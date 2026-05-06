/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "comm.h"
#include "net.h"
#include "graph.h"
#include "proxy.h"
#include "collectives.h"
#include "gdrwrap.h"
#include "shmutils.h"
#include "p2p.h"
#include "profiler.h"
#include "transport.h"
#include "shm.h"
#include "compiler.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include "register_inline.h"

static_assert(sizeof(ncclNetHandle_t) <= CONNECT_SIZE, "NET Connect info is too large");

#define NCCL_NET_MAP_HOSTMEM 0
#define NCCL_NET_MAP_DEVMEM 1
#define NCCL_NET_MAP_SHARED_HOSTMEM 2
#define NCCL_NET_MAP_SHARED_DEVMEM 3
#define NCCL_NET_MAP_GDCMEM 4
#define NCCL_NET_MAP_MEMS 5

#define NCCL_NET_MAP_MASK_DEVMEM 0x40000000
#define NCCL_NET_MAP_MASK_SHARED 0x80000000
#define NCCL_NET_MAP_MASK_USED   0x20000000
#define NCCL_NET_MAP_MASK_OFFSET 0x1fffffff

#define NCCL_NET_MAP_OFFSET_BANK(mapStruct, offsetName) \
  ((mapStruct)->offsets.offsetName >> 30)

#define NCCL_NET_MAP_OFFSET_NULL(mapStruct, offsetName) \
  (((mapStruct)->offsets.offsetName >> 29) == 0)

#define NCCL_NET_MAP_GET_POINTER(mapStruct, cpuOrGpu, offsetName) \
  (NCCL_NET_MAP_OFFSET_NULL(mapStruct, offsetName) ? NULL : \
   (mapStruct)->mems[NCCL_NET_MAP_OFFSET_BANK(mapStruct, offsetName)].cpuOrGpu##Ptr + ((mapStruct)->offsets.offsetName & NCCL_NET_MAP_MASK_OFFSET))

#define NCCL_NET_MAP_DEV_MEM(mapStruct, offsetName) \
  (((mapStruct)->offsets.offsetName & NCCL_NET_MAP_MASK_DEVMEM) != 0)

#define NCCL_NET_MAP_ADD_POINTER(mapStruct, shared, dev, memSize, offsetName) do { \
    int bank = NCCL_NET_MAP_MASK_USED + (dev)*NCCL_NET_MAP_MASK_DEVMEM + (shared)*NCCL_NET_MAP_MASK_SHARED; \
    if ((shared) == 0) { \
      if (dev) { \
        (mapStruct)->offsets.offsetName = bank + (mapStruct)->mems[NCCL_NET_MAP_DEVMEM].size; \
        (mapStruct)->mems[NCCL_NET_MAP_DEVMEM].size += memSize; \
      } else { \
        (mapStruct)->offsets.offsetName = bank + (mapStruct)->mems[NCCL_NET_MAP_HOSTMEM].size; \
        (mapStruct)->mems[NCCL_NET_MAP_HOSTMEM].size += memSize; \
      } \
    } else { \
      (mapStruct)->offsets.offsetName = bank; \
    } \
} while (0);

struct connectMapMem{
  char* gpuPtr;
  char* cpuPtr;
  int size;
  ncclIpcDesc ipcDesc;
  ncclShmIpcDesc_t attachDesc;
  ncclShmIpcDesc_t createDesc;
};

struct connectMap {
  int sameProcess;
  int shared;
  int cudaDev;
  // First 3 bits of offsets determine the mem bank. 001 is host mem, 011 is dev mem, 101 is shared host mem and 111 is shared dev mem.
  struct connectMapMem mems[NCCL_NET_MAP_MEMS];
  // Offsets. 3 MSBs indicate mem bank, 111 indicates NULL.
  struct {
    uint32_t sendMem;
    uint32_t recvMem;
    uint32_t buffs[NCCL_NUM_PROTOCOLS];
  } offsets;
};

struct sendNetResources {
  struct connectMap map;
  void* netSendComm;
  struct ncclSendMem* sendMem;
  struct ncclRecvMem* recvMem;

  int tpRank;
  int tpLocalRank;
  int tpRemoteRank;
  int netDev;
  enum ncclTopoGdrMode useGdr;
  int useDmaBuf;
  int maxRecvs;
  uint64_t* gdcSync;
  void* gdrDesc;
  int shared;
  int channelId;
  int connIndex;
  char* buffers[NCCL_NUM_PROTOCOLS];
  int buffSizes[NCCL_NUM_PROTOCOLS];
  void* mhandles[NCCL_NUM_PROTOCOLS];
  uint64_t step;
  uint64_t llLastCleaning;
  int netDeviceVersion;
  ncclNetDeviceType netDeviceType;
  ncclNetDeviceHandle_t* netDeviceHandle;
  size_t maxP2pBytes;
};

struct recvNetResources {
  struct connectMap map;
  void* netListenComm;
  void* netRecvComm;
  struct ncclSendMem* sendMem;
  struct ncclRecvMem* recvMem;

  int tpRank;
  int tpLocalRank;
  int tpRemoteRank;
  int tpRemoteProxyRank;
  int netDev;
  enum ncclTopoGdrMode useGdr;
  int useDmaBuf;
  int needFlush;
  int maxRecvs;
  uint64_t* gdcSync;
  uint64_t* gdcFlush;
  void* gdrDesc;
  int shared;
  int channelId;
  int connIndex;
  char* buffers[NCCL_NUM_PROTOCOLS];
  int buffSizes[NCCL_NUM_PROTOCOLS];
  void* mhandles[NCCL_NUM_PROTOCOLS];
  uint64_t step;
  uint64_t llLastCleaning;
  int netDeviceVersion;
  ncclNetDeviceType netDeviceType;
  ncclNetDeviceHandle_t* netDeviceHandle;
  size_t maxP2pBytes;
};

struct netRegInfo {
  uintptr_t buffer;
  size_t size;
  // Number of physical mapped segments that a buffer spans
  int numSegments;
};

/* Determine if two peers can communicate with NET */
static ncclResult_t canConnect(int* ret, struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclPeerInfo* info1, struct ncclPeerInfo* info2) {
  *ret = 1;
  if (info1->hostHash == info2->hostHash) {
    // If on the same host, check intra-node net is not disabled.
    NCCLCHECK(ncclTopoCheckNet(comm->topo, info1->rank, info2->rank, ret));
  }
  return ncclSuccess;
}

NCCL_PARAM(NetSharedBuffers, "NET_SHARED_BUFFERS", -2);
NCCL_PARAM(NetSharedComms, "NET_SHARED_COMMS", 1);
NCCL_PARAM(Phase0Log, "PHASE0_LOG", 0);
NCCL_PARAM(Phase1StaticW, "PHASE1_STATIC_W", 0);
NCCL_PARAM(Phase2B2Enable, "PHASE2_B2_ENABLE", 0);
NCCL_PARAM(Phase2Log, "PHASE2_LOG", 0);
NCCL_PARAM(Phase3B3Enable, "PHASE3_B3_ENABLE", 0);
NCCL_PARAM(Phase3Log, "PHASE3_LOG", 0);
NCCL_PARAM(Phase4Enable, "PHASE4_ENABLE", 0);
NCCL_PARAM(Phase4Log, "PHASE4_LOG", 0);
NCCL_PARAM(Phase4PostReceiveW, "PHASE4_POST_RECEIVE_W", 0);
NCCL_PARAM(Phase5Log, "PHASE5_LOG", 0);
NCCL_PARAM(Phase5ProgressLogEvery, "PHASE5_PROGRESS_LOG_EVERY", 128);
NCCL_PARAM(Phase6Enable, "PHASE6_ENABLE", 0);
NCCL_PARAM(Phase6Log, "PHASE6_LOG", 0);
NCCL_PARAM(Phase7Enable, "PHASE7_ENABLE", 0);
NCCL_PARAM(Phase7Log, "PHASE7_LOG", 0);
NCCL_PARAM(Phase9Log, "PHASE9_LOG", 0);
NCCL_PARAM(Appendix2GroupLog, "APPENDIX2_GROUP_LOG", 0);
NCCL_PARAM(Appendix2DisableWstallLog, "APPENDIX2_DISABLE_WSTALL_LOG", 0);
NCCL_PARAM(Phase3WarmupIntervals, "PHASE3_WARMUP_INTERVALS", 4);
NCCL_PARAM(Phase3HiIntervals, "PHASE3_HI_INTERVALS", 2);
NCCL_PARAM(Phase3LoIntervals, "PHASE3_LO_INTERVALS", 8);
NCCL_PARAM(Phase3OccRatioHighPct, "PHASE3_OCC_RATIO_HIGH_PCT", 90);
NCCL_PARAM(Phase3LagRatioHighPct, "PHASE3_LAG_RATIO_HIGH_PCT", 100);
NCCL_PARAM(Phase3DelayRatioHighPct, "PHASE3_DELAY_RATIO_HIGH_PCT", 125);

struct setupReq {
  int tpRank;
  int tpLocalRank;
  int tpRemoteRank;
  int shared;
  int netDev;
  enum ncclTopoGdrMode useGdr;
  int needFlush;
  int channelId;
  int connIndex;
};

NCCL_PARAM(NetOptionalRecvCompletion, "NET_OPTIONAL_RECV_COMPLETION", 1);

static inline void phase0ProxyLog(
    struct ncclProxyState* proxyState,
    struct ncclProxyArgs* args,
    struct ncclProxySubArgs* sub,
    const char* event,
    int slot,
    ssize_t size,
    int wBase,
    int wCfg,
    int wEff) {
  if (ncclParamPhase0Log() == 0) return;
  INFO(NCCL_NET,
      "PHASE0 event=%s tNs=%llu rank=%d peer=%d channel=%d slot=%d coll=%s collApi=%s algo=%s proto=%s size=%lld base=%llu posted=%llu received=%llu transmitted=%llu done=%llu nsteps=%d wBase=%d wCfg=%d wEff=%d occPd=%llu occTr=%llu",
      event,
      (unsigned long long)clockNano(),
      proxyState->tpRank,
      sub->peer,
      sub->channelId,
      slot,
      ncclFuncToString((ncclFunc_t)args->coll),
      ncclFuncToString((ncclFunc_t)args->collAPI),
      ncclAlgoToString(args->algorithm),
      ncclProtoToString(args->protocol),
      (long long)size,
      (unsigned long long)sub->base,
      (unsigned long long)sub->posted,
      (unsigned long long)sub->received,
      (unsigned long long)sub->transmitted,
      (unsigned long long)sub->done,
      sub->nsteps,
      wBase,
      wCfg,
      wEff,
      (unsigned long long)(sub->posted - sub->done),
      (unsigned long long)(sub->transmitted - sub->done));
}

static inline int phase5ShouldLogProgress(uint64_t callCount) {
  int logEvery = ncclParamPhase5ProgressLogEvery();
  return ncclParamPhase5Log() != 0 && logEvery > 0 && (callCount == 1 || (callCount % (uint64_t)logEvery) == 0);
}

static inline void phase5RecvProgressLog(
    struct ncclProxyState* proxyState,
    struct ncclProxyArgs* args,
    int maxDepth,
    int phase4Enabled,
    double phase4WRaw,
    uint64_t deltaNs) {
  if (!phase5ShouldLogProgress(args->phase5RecvProxyCalls)) return;
  INFO(NCCL_NET,
      "PHASE5 event=RECV_PROXY_PROGRESS tNs=%llu rank=%d call=%llu deltaNs=%llu state=%d idle=%d done=%d nsubs=%d coll=%s collApi=%s algo=%s proto=%s maxDepth=%d phase4Enable=%d phase4WRaw=%.3f",
      (unsigned long long)clockNano(),
      proxyState->tpRank,
      (unsigned long long)args->phase5RecvProxyCalls,
      (unsigned long long)deltaNs,
      args->state,
      args->idle,
      args->done,
      args->nsubs,
      ncclFuncToString((ncclFunc_t)args->coll),
      ncclFuncToString((ncclFunc_t)args->collAPI),
      ncclAlgoToString(args->algorithm),
      ncclProtoToString(args->protocol),
      maxDepth,
      phase4Enabled,
      phase4WRaw);
}

static inline void phase5RecvEventLog(
    struct ncclProxyState* proxyState,
    struct ncclProxyArgs* args,
    struct ncclProxySubArgs* sub,
    const char* event,
    int slot,
    ssize_t size,
    int maxDepth,
    double phase4WRaw,
    int wEff,
    uint64_t postTsNs,
    uint64_t postToNetDoneNs,
    uint64_t progressCallsSincePost) {
  if (ncclParamPhase5Log() == 0) return;
  INFO(NCCL_NET,
      "PHASE5 event=%s tNs=%llu rank=%d peer=%d channel=%d slot=%d coll=%s collApi=%s algo=%s proto=%s size=%lld base=%llu posted=%llu received=%llu transmitted=%llu done=%llu nsteps=%d maxDepth=%d wCfgRaw=%.3f wEff=%d postTsNs=%llu postToNetDoneNs=%llu progressCallsSincePost=%llu occPr=%llu occPd=%llu",
      event,
      (unsigned long long)clockNano(),
      proxyState->tpRank,
      sub->peer,
      sub->channelId,
      slot,
      ncclFuncToString((ncclFunc_t)args->coll),
      ncclFuncToString((ncclFunc_t)args->collAPI),
      ncclAlgoToString(args->algorithm),
      ncclProtoToString(args->protocol),
      (long long)size,
      (unsigned long long)sub->base,
      (unsigned long long)sub->posted,
      (unsigned long long)sub->received,
      (unsigned long long)sub->transmitted,
      (unsigned long long)sub->done,
      sub->nsteps,
      maxDepth,
      phase4WRaw,
      wEff,
      (unsigned long long)postTsNs,
      (unsigned long long)postToNetDoneNs,
      (unsigned long long)progressCallsSincePost,
      (unsigned long long)(sub->posted - sub->received),
      (unsigned long long)(sub->posted - sub->done));
}

static_assert(sizeof(ncclNetHandle_t) + sizeof(int) <= CONNECT_SIZE, "Not large enough ncclConnect to hold ncclNetHandle_t and useGdr flag");

// Common function to initialize network attributes from a ncclComm
static void populateCommNetAttrs(struct ncclComm* comm, struct ncclConnector* conn, ncclNetAttr_t* netAttr) {
  *netAttr = NCCL_NET_ATTR_INIT;
  netAttr->sendCommAttr.minConcurrentPeers = 1;
  netAttr->sendCommAttr.minFlowsPerPeer = 1;

  netAttr->recvCommAttr.minConcurrentPeers = 1;
  netAttr->recvCommAttr.minFlowsPerPeer = 1;

  if (conn->p2pOnly) {
    size_t maxConcPeers = comm->p2pnChannels * NCCL_MAX_DEV_WORK_P2P_PER_BATCH;
    if (comm->nRanks < maxConcPeers) maxConcPeers = comm->nRanks;

    netAttr->sendCommAttr.maxConcurrentPeers = maxConcPeers;
    netAttr->sendCommAttr.maxFlowsPerPeer = comm->p2pnChannelsPerPeer;
    netAttr->recvCommAttr.maxConcurrentPeers = maxConcPeers;
    netAttr->recvCommAttr.maxFlowsPerPeer = comm->p2pnChannelsPerPeer;
    netAttr->op = BIT(ncclFuncSend) | BIT(ncclFuncRecv) |
                  BIT(ncclFuncAlltoAll) | BIT(ncclFuncScatter) | BIT(ncclFuncGather);
  } else {
    size_t maxConcPeers = (NCCL_MAX_TREE_ARITY - 1) * 2;
    if (comm->nRanks < maxConcPeers) maxConcPeers = comm->nRanks;
    netAttr->sendCommAttr.maxConcurrentPeers = maxConcPeers;
    netAttr->sendCommAttr.maxFlowsPerPeer = comm->nChannels;
    netAttr->recvCommAttr.maxConcurrentPeers = maxConcPeers;
    netAttr->recvCommAttr.maxFlowsPerPeer = comm->nChannels;
  }
}

// Apply the netAttr to the netComm
void setNetAttrs(struct ncclProxyState* proxyState, ncclNetAttr_t* netAttr)
{
  if (proxyState->ncclNet->setNetAttr) {
    proxyState->ncclNet->setNetAttr(proxyState->netContext, netAttr);
    proxyState->netAttr = *netAttr;
  }
}

void printNetAttrs(ncclNetAttr_t* netAttr, const char *task)
{
  const int opBufLen = ncclNumFuncs*32;
  char opBuf[opBufLen] = "";
  const int algoBufLen = NCCL_NUM_ALGORITHMS*32;
  char algoBuf[algoBufLen] = "";
  const int protoBufLen = NCCL_NUM_PROTOCOLS*32;
  char protoBuf[protoBufLen] = "";

  ncclBitsToString(netAttr->op, MASK(ncclNumFuncs), (const char* (*)(int))ncclFuncToString, opBuf, opBufLen, "*");
  ncclBitsToString(netAttr->algo, MASK(NCCL_NUM_ALGORITHMS), ncclAlgoToString, algoBuf, algoBufLen, "*");
  ncclBitsToString(netAttr->proto, MASK(NCCL_NUM_PROTOCOLS), ncclProtoToString, protoBuf, protoBufLen, "*");

  TRACE(NCCL_NET, "%s hints, send peers/flows: [%d-%d][%d-%d] recv peers/flows: [%d-%d][%d-%d] op: %s algo: %s proto: %s",
        task, netAttr->sendCommAttr.minConcurrentPeers, netAttr->sendCommAttr.maxConcurrentPeers,
        netAttr->sendCommAttr.minFlowsPerPeer, netAttr->sendCommAttr.maxFlowsPerPeer,
        netAttr->recvCommAttr.minConcurrentPeers, netAttr->recvCommAttr.maxConcurrentPeers,
        netAttr->recvCommAttr.minFlowsPerPeer, netAttr->recvCommAttr.maxFlowsPerPeer,
        opBuf, algoBuf, protoBuf);
}

// Set the netAttr for a transfer operation
void setXferNetAttrs(struct ncclProxyState* proxyState, struct ncclProxyArgs* args, int send)
{
  ncclNetAttr_t netAttr;

  if (!proxyState->ncclNet->setNetAttr)
    return;

  netAttr = proxyState->netAttr;

  if (send) {
    netAttr.sendCommAttr.maxConcurrentPeers = args->nPeers;
    netAttr.sendCommAttr.minConcurrentPeers = args->nPeers;
    netAttr.sendCommAttr.maxFlowsPerPeer = args->nChannels;
    netAttr.sendCommAttr.minFlowsPerPeer = args->nChannels;
  } else {
    netAttr.recvCommAttr.maxConcurrentPeers = args->nPeers;
    netAttr.recvCommAttr.minConcurrentPeers = args->nPeers;
    netAttr.recvCommAttr.maxFlowsPerPeer = args->nChannels;
    netAttr.recvCommAttr.minFlowsPerPeer = args->nChannels;
  }

  netAttr.op = BIT(args->collAPI);
  // algo/proto are undefined for p2p
  if (args->collAPI < NCCL_NUM_FUNCTIONS) {
    netAttr.algo = BIT(args->algorithm);
    netAttr.proto = BIT(args->protocol);
  }

  if (memcmp(&proxyState->netAttr, &netAttr, sizeof(netAttr))) {
    setNetAttrs(proxyState, &netAttr);
    printNetAttrs(&netAttr, send ? "send" : "recv");
  }
}

// Forward declaration
static ncclResult_t sendProxyProgress(struct ncclProxyState* proxyState, struct ncclProxyArgs* args);

// Returns the flags to be used by a call to cuMemGetHandleForAddressRange.
static inline int getHandleForAddressRangeFlags(ncclTopoGdrMode useGdr) {
  int flags = 0;
#if CUDA_VERSION >= 12080
  // Force mapping on PCIe on systems with both PCI and C2C attachments.
  if (useGdr == ncclTopoGdrModePci) flags = CU_MEM_RANGE_FLAG_DMA_BUF_MAPPING_TYPE_PCIE;
#endif
  return flags;
}

/* Determine if we will use this transport for this peer and return connect
 * information for this peer */
static ncclResult_t sendSetup(struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo, struct ncclConnect* connectInfo, struct ncclConnector* send, int channelId, int connIndex) {
  struct setupReq req = { 0 };

  send->conn.shared = req.shared = graph || connIndex == 0 ? 0 : ncclParamNetSharedBuffers() != -2 ? ncclParamNetSharedBuffers() : 1;
  req.channelId = channelId;
  req.connIndex = connIndex;

  int proxyRank;
  int64_t netId;
  NCCLCHECK(ncclTopoGetNetDev(comm, myInfo->rank, graph, channelId, peerInfo->rank, &netId, &req.netDev, &proxyRank));
  NCCLCHECK(ncclTopoCheckGdr(comm->topo, myInfo->rank, netId, 1, &req.useGdr));
  send->conn.flags |= req.useGdr ? NCCL_DIRECT_NIC : 0;
  if (!req.useGdr && connIndex == 0) comm->useGdr = 0;
  if (proxyRank != myInfo->rank && connIndex == 0) comm->useNetPXN = true;

  NCCLCHECK(ncclProxyConnect(comm, TRANSPORT_NET, 1, proxyRank, &send->proxyConn));
  req.tpLocalRank = comm->topParentLocalRanks[comm->localRank];
  req.tpRank = comm->topParentRanks[myInfo->rank];
  req.tpRemoteRank = comm->topParentRanks[peerInfo->rank];
  NCCLCHECK(ncclProxyCallBlocking(comm, &send->proxyConn, ncclProxyMsgSetup, &req, sizeof(req), NULL, 0));

  if (proxyRank == myInfo->rank) {
    INFO(NCCL_INIT|NCCL_NET,"Channel %02d/%d : %d[%d] -> %d[%d] [send] via NET/%s/%d%s%s%s", channelId, connIndex, myInfo->rank, myInfo->nvmlDev, peerInfo->rank, peerInfo->nvmlDev, comm->ncclNet->name, req.netDev,
        req.useGdr ? "/GDRDMA" : "", req.useGdr==ncclTopoGdrModePci ? "(PCI)" : "",
        req.shared ? "/Shared" : "");
  } else {
    INFO(NCCL_INIT|NCCL_NET,"Channel %02d/%d : %d[%d] -> %d[%d] [send] via NET/%s/%d(%d)%s%s%s", channelId, connIndex, myInfo->rank, myInfo->nvmlDev, peerInfo->rank, peerInfo->nvmlDev, comm->ncclNet->name, req.netDev,
        proxyRank,
        req.useGdr ? "/GDRDMA" : "", req.useGdr==ncclTopoGdrModePci ? "(PCI)" : "",
        req.shared ? "/Shared" : "");
  }
  *((int*)connectInfo) = comm->topParentRanks[proxyRank];
  memcpy((uint8_t*)connectInfo + sizeof(ncclNetHandle_t), &req.useGdr, sizeof(int));
  return ncclSuccess;
}

// GDRCOPY support: TAIL_ENABLE When enabled locates the RX proxy tail in CUDA memory
NCCL_PARAM(GdrCopySyncEnable, "GDRCOPY_SYNC_ENABLE", 1);
// GDRCOPY support: FLUSH_ENABLE When enabled uses a PCI-E read to flush GDRDMA buffers
NCCL_PARAM(GdrCopyFlushEnable, "GDRCOPY_FLUSH_ENABLE", 0);

/* Setup recv connector */
static ncclResult_t recvSetup(struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo, struct ncclConnect* connectInfo, struct ncclConnector* recv, int channelId, int connIndex) {
  struct setupReq req = { 0 };

  recv->conn.shared = req.shared = graph || connIndex == 0 ? 0 : ncclParamNetSharedBuffers() != -2 ? ncclParamNetSharedBuffers() : 1;
  req.channelId = channelId;
  req.connIndex = connIndex;

  // Use myInfo->rank as the receiver uses its own NIC
  int proxyRank;
  int64_t netId;
  NCCLCHECK(ncclTopoGetNetDev(comm, myInfo->rank, graph, channelId, myInfo->rank, &netId, &req.netDev, &proxyRank));
  NCCLCHECK(ncclTopoCheckGdr(comm->topo, myInfo->rank, netId, 0, &req.useGdr));
  recv->conn.flags |= req.useGdr ? NCCL_DIRECT_NIC : 0;
  if (!req.useGdr && connIndex == 0) comm->useGdr = 0;

  // Determine whether we need to flush the GDR buffer on recv or not
  if (req.useGdr) NCCLCHECK(ncclTopoNeedFlush(comm, netId, req.netDev, myInfo->rank, &req.needFlush));

  // We don't support PXN on receive yet
  NCCLCHECK(ncclProxyConnect(comm, TRANSPORT_NET, 0, myInfo->rank, &recv->proxyConn));

  req.tpLocalRank = comm->topParentLocalRanks[comm->localRank];
  req.tpRank = comm->topParentRanks[myInfo->rank];
  req.tpRemoteRank = comm->topParentRanks[peerInfo->rank];
  NCCLCHECK(ncclProxyCallBlocking(comm, &recv->proxyConn, ncclProxyMsgSetup, &req, sizeof(req), connectInfo, sizeof(ncclNetHandle_t)));
  memcpy((uint8_t*)connectInfo + sizeof(ncclNetHandle_t), &req.useGdr, sizeof(int));
  INFO(NCCL_INIT|NCCL_NET,"Channel %02d/%d : %d[%d] -> %d[%d] [receive] via NET/%s/%d%s%s%s", channelId, connIndex, peerInfo->rank, peerInfo->nvmlDev, myInfo->rank, myInfo->nvmlDev, comm->ncclNet->name, req.netDev,
      req.useGdr ? "/GDRDMA" : "", req.useGdr==ncclTopoGdrModePci ? "(PCI)" : "",
      req.shared ? "/Shared" : "");
  return ncclSuccess;
}

static ncclResult_t netMapShm(struct ncclComm *comm, struct ncclProxyConnector* proxyConn, struct connectMapMem* mem) {
  NCCLCHECK(ncclShmImportShareableBuffer(comm, proxyConn->rank, &mem->createDesc, (void**)&mem->cpuPtr, (void**)&mem->gpuPtr, &mem->attachDesc));
  return ncclSuccess;
}

static ncclResult_t netCreateShm(struct ncclProxyState* proxyState, struct connectMapMem* mem) {
  NCCLCHECK(ncclShmAllocateShareableBuffer(mem->size, false, &mem->createDesc, (void**)&mem->cpuPtr, (void**)&mem->gpuPtr));
  return ncclSuccess;
}

static ncclResult_t netDumpMap(struct connectMap* map) {
  printf("Dump map same process %d shared %d\n", map->sameProcess, map->shared);
  struct connectMapMem *mem = map->mems+NCCL_NET_MAP_HOSTMEM;
  printf("Mem 0: Host mem (%x B) CPU %p GPU %p\n", mem->size, mem->cpuPtr, mem->gpuPtr);
  mem = map->mems+NCCL_NET_MAP_DEVMEM;
  printf("Mem 1: Vid  mem (%x B) CPU %p GPU %p\n", mem->size, mem->cpuPtr, mem->gpuPtr);
  mem = map->mems+NCCL_NET_MAP_SHARED_HOSTMEM;
  printf("Mem 2: Shared Host mem (%x B) CPU %p GPU %p\n", mem->size, mem->cpuPtr, mem->gpuPtr);
  mem = map->mems+NCCL_NET_MAP_SHARED_DEVMEM;
  printf("Mem 3: Shared Vid mem (%x B) CPU %p GPU %p\n", mem->size, mem->cpuPtr, mem->gpuPtr);
  printf("SendMem -> Used %d Bank %d Offset %x, cpu %p gpu %p\n",
      map->offsets.sendMem & NCCL_NET_MAP_MASK_USED ? 1 : 0,
      NCCL_NET_MAP_OFFSET_BANK(map, sendMem), map->offsets.sendMem & NCCL_NET_MAP_MASK_OFFSET,
      NCCL_NET_MAP_GET_POINTER(map, cpu, sendMem), NCCL_NET_MAP_GET_POINTER(map, gpu, sendMem));
  printf("RecvMem -> Used %d Bank %d Offset %x, cpu %p gpu %p\n",
      map->offsets.recvMem & NCCL_NET_MAP_MASK_USED ? 1 : 0,
      NCCL_NET_MAP_OFFSET_BANK(map, recvMem), map->offsets.recvMem & NCCL_NET_MAP_MASK_OFFSET,
      NCCL_NET_MAP_GET_POINTER(map, cpu, recvMem), NCCL_NET_MAP_GET_POINTER(map, gpu, recvMem));
  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
    printf("Proto %d -> Used %d Bank %d Offset %x, cpu %p, gpu %p\n", p,
        map->offsets.buffs[p] & NCCL_NET_MAP_MASK_USED ? 1 : 0,
        NCCL_NET_MAP_OFFSET_BANK(map, buffs[p]), map->offsets.buffs[p] & NCCL_NET_MAP_MASK_OFFSET,
        NCCL_NET_MAP_GET_POINTER(map, cpu, buffs[p]), NCCL_NET_MAP_GET_POINTER(map, gpu, buffs[p]));
  }
  printf("End of dump\n");
  return ncclSuccess;
}

struct netSendConnectArgs {
  ncclNetHandle_t handle;
  ncclNetAttr_t netAttr;
};

struct netRecvConnectArgs {
  int proxyRank;
  ncclNetAttr_t netAttr;
};

static ncclResult_t sendConnect(struct ncclComm* comm, struct ncclConnect* connectInfo, int nranks, int rank, struct ncclConnector* send) {
  struct connectMap* map = (connectMap*) send->transportResources;
  void* opId;
  int recvUseGdr;

  memcpy(&recvUseGdr, (uint8_t*)connectInfo + sizeof(ncclNetHandle_t), sizeof(int));
  if (!recvUseGdr) send->conn.flags &= ~NCCL_DIRECT_NIC;

  // map isn't allocated thus this op hasn't been submitted yet
  if (!map) {
    // Setup device pointers
    NCCLCHECK(ncclCalloc(&map, 1));
    send->transportResources = map;
    opId = send;
    INFO(NCCL_PROXY, "sendConnect ncclProxyCallAsync opId=%p", opId);
    netSendConnectArgs args = {0};
    memcpy(&args.handle, connectInfo, sizeof(ncclNetHandle_t));

    populateCommNetAttrs(comm, send, &args.netAttr);

    NCCLCHECK(ncclProxyCallAsync(comm, &send->proxyConn, ncclProxyMsgConnect, &args, sizeof(netSendConnectArgs), sizeof(struct connectMap), opId));
  } else {
    opId =  send;
  }

  ncclResult_t ret;
  ret = ncclPollProxyResponse(comm, &send->proxyConn, map, opId);
  if (ret != ncclSuccess) {
    if (ret != ncclInProgress) {
      free(map);
      send->transportResources = NULL;
    }
    return ret;
  }
  INFO(NCCL_PROXY, "sendConnect ncclPollProxyResponse opId=%p", opId);

  if (map->sameProcess && !ncclCuMemEnable()) {
    if (map->cudaDev != comm->cudaDev) {
      // Enable P2P access for Legacy IPC
      cudaError_t err = cudaDeviceEnablePeerAccess(map->cudaDev, 0);
      if (err == cudaErrorPeerAccessAlreadyEnabled) {
        cudaGetLastError();
      } else if (err != cudaSuccess) {
        WARN("failed to peer with device %d: %d %s", map->cudaDev, err, cudaGetErrorString(err));
        return ncclInternalError;
      }
    }
  } else if (!(map->sameProcess && map->cudaDev == comm->cudaDev)) {
    if (!map->sameProcess) NCCLCHECK(netMapShm(comm, &send->proxyConn, map->mems + NCCL_NET_MAP_HOSTMEM));
    if (map->mems[NCCL_NET_MAP_DEVMEM].size) {
      map->mems[NCCL_NET_MAP_DEVMEM].gpuPtr = NULL;
      // NET transport import: No ownerVA available, mark as Persist (do not release)
      NCCLCHECK(ncclP2pImportShareableBuffer(comm, send->proxyConn.rank,
                                             map->mems[NCCL_NET_MAP_DEVMEM].size,
                                             &map->mems[NCCL_NET_MAP_DEVMEM].ipcDesc,
                                             (void**)&map->mems[NCCL_NET_MAP_DEVMEM].gpuPtr,
                                             nullptr, ncclMemPersist));
      map->mems[NCCL_NET_MAP_DEVMEM].cpuPtr = NULL;
    }
    if (map->mems[NCCL_NET_MAP_SHARED_DEVMEM].size) {
      void** sharedDevMemPtr = comm->proxyState->sharedDevMems + send->proxyConn.tpLocalRank;
      if (*sharedDevMemPtr == NULL) {
        map->mems[NCCL_NET_MAP_SHARED_DEVMEM].gpuPtr = NULL;
        // NET transport shared import: No ownerVA, mark as Persist (do not release)
        NCCLCHECK(ncclP2pImportShareableBuffer(comm, send->proxyConn.rank,
                                               map->mems[NCCL_NET_MAP_SHARED_DEVMEM].size,
                                               &map->mems[NCCL_NET_MAP_SHARED_DEVMEM].ipcDesc,
                                               sharedDevMemPtr,
                                               nullptr, ncclMemPersist));
      }
      map->mems[NCCL_NET_MAP_SHARED_DEVMEM].gpuPtr = (char*)(*sharedDevMemPtr);
      map->mems[NCCL_NET_MAP_SHARED_DEVMEM].cpuPtr = NULL;
    }
  }
  //NCCLCHECK(netDumpMap(map));

  struct ncclSendMem *sendMem = (struct ncclSendMem*) NCCL_NET_MAP_GET_POINTER(map, gpu, sendMem);
  void* gdcMem = map->mems[NCCL_NET_MAP_GDCMEM].gpuPtr;
  send->conn.head = gdcMem ? (uint64_t*)gdcMem : &sendMem->head;

  struct ncclRecvMem *recvMem = (struct ncclRecvMem*) NCCL_NET_MAP_GET_POINTER(map, gpu, recvMem);
  send->conn.tail = &recvMem->tail;
  send->conn.stepSize = comm->buffSizes[NCCL_PROTO_SIMPLE]/NCCL_STEPS;
  send->conn.connFifo = recvMem->connFifo;
  // Only fuse P2P buffers, continue to allocate dedicated buffers for ring/tree
  for (int i=0; i<NCCL_STEPS; i++) {
    send->conn.connFifo[i].offset = -1;
    recvMem->connFifo[i].mode = map->shared ? NCCL_MODE_OFFSET : NCCL_MODE_NORMAL;
  }

  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++)
    send->conn.buffs[p] = NCCL_NET_MAP_GET_POINTER(map, gpu, buffs[p]);

  if (send->proxyConn.sameProcess) {
    if (send->proxyConn.connection->netDeviceHandle) {
      send->conn.netDeviceHandle = *send->proxyConn.connection->netDeviceHandle;

      for (int p=0; p<NCCL_NUM_PROTOCOLS; p++)
        send->conn.mhandles[p] = send->proxyConn.connection->mhandles[p];
    }

    if (send->proxyConn.connection->needsProxyProgress) {
      send->proxyConn.proxyProgress = sendProxyProgress;
    } else {
      send->proxyConn.proxyProgress = NULL;
    }
  } else {
    send->proxyConn.proxyProgress = sendProxyProgress;
  }

  return ncclSuccess;
}

// Forward declare
static ncclResult_t recvProxyProgress(struct ncclProxyState* proxyState, struct ncclProxyArgs* args);

/* Connect to this peer */
static ncclResult_t recvConnect(struct ncclComm* comm, struct ncclConnect* connectInfo, int nranks, int rank, struct ncclConnector* recv) {
  struct connectMap* map = (connectMap*) recv->transportResources;
  void* opId;
  int sendUseGdr;

  memcpy(&sendUseGdr, (uint8_t*)connectInfo + sizeof(ncclNetHandle_t), sizeof(int));
  if (!sendUseGdr) recv->conn.flags &= ~NCCL_DIRECT_NIC;

  if (!map) {
    NCCLCHECK(ncclCalloc(&map, 1));
    recv->transportResources = map;
    // Use recv connector as unique identifier
    opId = recv;
    INFO(NCCL_PROXY, "recvConnect ncclProxyCallAsync opId=%p &recv->proxyConn=%p connectInfo=%p",
       opId, &recv->proxyConn, connectInfo);
    netRecvConnectArgs args = {0};
    args.proxyRank = *((int*)connectInfo);

    populateCommNetAttrs(comm, recv, &args.netAttr);

    NCCLCHECK(ncclProxyCallAsync(comm, &recv->proxyConn, ncclProxyMsgConnect, &args, sizeof(netRecvConnectArgs), sizeof(struct connectMap), opId));
  } else {
    opId = recv;
  }

  ncclResult_t ret;
  NCCLCHECK(ret = ncclPollProxyResponse(comm, &recv->proxyConn, map, opId));
  if (ret != ncclSuccess) {
    if (ret != ncclInProgress) {
      free(map);
      recv->transportResources = NULL;
    }
    return ret;
  }
  INFO(NCCL_PROXY, "recvConnect ncclPollProxyResponse opId=%p", opId);
  //NCCLCHECK(netDumpMap(map));

  struct ncclSendMem *sendMem = (struct ncclSendMem*) NCCL_NET_MAP_GET_POINTER(map, gpu, sendMem);
  recv->conn.head = &sendMem->head;

  struct ncclRecvMem *recvMem = (struct ncclRecvMem*) NCCL_NET_MAP_GET_POINTER(map, gpu, recvMem);
  void* gdcMem = map->mems[NCCL_NET_MAP_GDCMEM].gpuPtr;
  recv->conn.tail = gdcMem ? (uint64_t*)gdcMem : &recvMem->tail;
  recv->conn.stepSize = comm->buffSizes[NCCL_PROTO_SIMPLE]/NCCL_STEPS;
  recv->conn.connFifo = recvMem->connFifo;
  // Only fuse P2P buffers, continue to allocate dedicated buffers for ring/tree
  for (int i=0; i<NCCL_STEPS; i++) {
    recvMem->connFifo[i].mode = map->shared ? NCCL_MODE_OFFSET : NCCL_MODE_NORMAL;
  }

  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++)
    recv->conn.buffs[p] = NCCL_NET_MAP_GET_POINTER(map, gpu, buffs[p]);

  if (recv->proxyConn.sameProcess) {
    if (recv->proxyConn.connection->netDeviceHandle) {
      recv->conn.netDeviceHandle = *recv->proxyConn.connection->netDeviceHandle;

      for (int p=0; p<NCCL_NUM_PROTOCOLS; p++)
        recv->conn.mhandles[p] = recv->proxyConn.connection->mhandles[p];
    }

    if (recv->proxyConn.connection->needsProxyProgress) {
      recv->proxyConn.proxyProgress = recvProxyProgress;
    } else {
      recv->proxyConn.proxyProgress = NULL;
    }
  } else {
    recv->proxyConn.proxyProgress = recvProxyProgress;
  }

  return ncclSuccess;
}

static ncclResult_t sendFree(struct ncclComm* comm, struct ncclConnector* send) {
  struct connectMap* map = (struct connectMap*)(send->transportResources);
  if (map) {
    int cudaDev;
    CUDACHECK(cudaGetDevice(&cudaDev));
    if (map->cudaDev != cudaDev && map->mems[NCCL_NET_MAP_DEVMEM].size) {
      if (ncclCuMemEnable()) {
        // cuMem API support
        NCCLCHECK(ncclP2pFreeShareableBuffer(&map->mems[NCCL_NET_MAP_DEVMEM].ipcDesc));
        NCCLCHECK(ncclCuMemFree(map->mems[NCCL_NET_MAP_DEVMEM].gpuPtr, comm->memManager));
      } else {
        // Legacy CUDA IPC support
        CUDACHECK(cudaIpcCloseMemHandle(map->mems[NCCL_NET_MAP_DEVMEM].gpuPtr));
      }
    }
    if (!map->sameProcess) {
      NCCLCHECK(ncclShmIpcClose(&map->mems[NCCL_NET_MAP_HOSTMEM].attachDesc));
    }
    free(map);
  }

  return ncclSuccess;
}

static ncclResult_t recvFree(struct ncclComm* comm, struct ncclConnector* recv) {
  if (recv->transportResources) free(recv->transportResources);
  return ncclSuccess;
}

#define NCCL_SHARED_STEPS 16

static inline int phase1WindowBaseDepth(struct ncclProxyArgs* args) {
  return std::min(NCCL_STEPS, NCCL_SHARED_STEPS/args->nsubs);
}

static inline int phase1WindowCfg() {
  return ncclParamPhase1StaticW();
}

static inline int phase1WindowEff(struct ncclProxyArgs* args) {
  int wBase = phase1WindowBaseDepth(args);
  int wCfg = phase1WindowCfg();
  return (wCfg > 0) ? std::min(wBase, std::max(1, wCfg)) : wBase;
}

static inline double phase4WindowRaw() {
  const char* env = getenv("NCCL_PHASE4_POST_RECEIVE_W");
  if (env == NULL || env[0] == '\0') return 0.0;
  char* end = NULL;
  double value = strtod(env, &end);
  if (end == env) return 0.0;
  if (value < 0.0) return 0.0;
  return value;
}

static inline double phase6PostRateRaw() {
  const char* env = getenv("NCCL_PHASE6_POST_RATE");
  if (env == NULL || env[0] == '\0') return 0.0;
  char* end = NULL;
  double value = strtod(env, &end);
  if (end == env) return 0.0;
  if (value < 0.0) return 0.0;
  return value;
}

static inline double phase6PostBurstRaw() {
  const char* env = getenv("NCCL_PHASE6_POST_BURST");
  if (env == NULL || env[0] == '\0') return 0.0;
  char* end = NULL;
  double value = strtod(env, &end);
  if (end == env) return 0.0;
  if (value < 0.0) return 0.0;
  return value;
}

static inline int phase6Enabled() {
  return ncclParamPhase6Enable() != 0 || phase6PostRateRaw() > 0.0;
}

static inline double phase7RateRatioRaw() {
  const char* env = getenv("NCCL_PHASE7_POST_RATE_RATIO_PCT");
  if (env == NULL || env[0] == '\0') return 0.0;
  char* end = NULL;
  double value = strtod(env, &end);
  if (end == env) return 0.0;
  if (value < 0.0) return 0.0;
  return value;
}

static inline double phase7ObserveMsRaw() {
  const char* env = getenv("NCCL_PHASE7_OBSERVE_MS");
  if (env == NULL || env[0] == '\0') return 50.0;
  char* end = NULL;
  double value = strtod(env, &end);
  if (end == env) return 50.0;
  return value > 0.0 ? value : 50.0;
}

static inline double phase7BurstWindowMsRaw() {
  const char* env = getenv("NCCL_PHASE7_BURST_WINDOW_MS");
  if (env == NULL || env[0] == '\0') return 4.0;
  char* end = NULL;
  double value = strtod(env, &end);
  if (end == env) return 4.0;
  return value > 0.0 ? value : 4.0;
}

static inline double phase7BurstFloorPostsRaw() {
  const char* env = getenv("NCCL_PHASE7_BURST_FLOOR_POSTS");
  if (env == NULL || env[0] == '\0') return 4.0;
  char* end = NULL;
  double value = strtod(env, &end);
  if (end == env) return 4.0;
  return value > 0.0 ? value : 4.0;
}

static inline int phase7Enabled() {
  return ncclParamPhase7Enable() != 0 || phase7RateRatioRaw() > 0.0;
}

static inline void phase6RateCfgLog(
    struct ncclProxyState* proxyState,
    struct ncclProxyArgs* args,
    struct ncclProxySubArgs* sub,
    double rateRaw,
    double burstRaw) {
  if (ncclParamPhase6Log() == 0) return;
  INFO(NCCL_NET,
      "PHASE6 event=RATE_CFG tNs=%llu rank=%d peer=%d channel=%d groupSize=%d coll=%s collApi=%s algo=%s proto=%s postRatePerMs=%.6f postBurst=%.3f",
      (unsigned long long)clockNano(),
      proxyState->tpRank,
      sub->peer,
      sub->channelId,
      sub->groupSize,
      ncclFuncToString((ncclFunc_t)args->coll),
      ncclFuncToString((ncclFunc_t)args->collAPI),
      ncclAlgoToString(args->algorithm),
      ncclProtoToString(args->protocol),
      rateRaw,
      burstRaw);
}

static inline void phase6RateDecisionLog(
    struct ncclProxyState* proxyState,
    struct ncclProxyArgs* args,
    struct ncclProxySubArgs* sub,
    const char* event,
    uint64_t elapsedNs,
    int postCost,
    double rateRaw,
    double burstRaw,
    double tokensBefore,
    double tokensAfter) {
  if (ncclParamPhase6Log() == 0) return;
  INFO(NCCL_NET,
      "PHASE6 event=%s tNs=%llu rank=%d peer=%d channel=%d groupSize=%d coll=%s collApi=%s algo=%s proto=%s elapsedNs=%llu postCost=%d postRatePerMs=%.6f postBurst=%.3f tokensBefore=%.6f tokensAfter=%.6f",
      event,
      (unsigned long long)clockNano(),
      proxyState->tpRank,
      sub->peer,
      sub->channelId,
      sub->groupSize,
      ncclFuncToString((ncclFunc_t)args->coll),
      ncclFuncToString((ncclFunc_t)args->collAPI),
      ncclAlgoToString(args->algorithm),
      ncclProtoToString(args->protocol),
      (unsigned long long)elapsedNs,
      postCost,
      rateRaw,
      burstRaw,
      tokensBefore,
      tokensAfter);
}

static inline void phase7RateCfgLog(
    struct ncclProxyState* proxyState,
    struct ncclProxyArgs* args,
    struct ncclProxySubArgs* sub,
    double ratioPct,
    double observeMs,
    double burstWindowMs,
    double burstFloorPosts) {
  if (ncclParamPhase7Log() == 0) return;
  INFO(NCCL_NET,
      "PHASE7 event=RATE_CFG tNs=%llu rank=%d peer=%d channel=%d groupSize=%d coll=%s collApi=%s algo=%s proto=%s ratioPct=%.3f observeMs=%.3f burstWindowMs=%.3f burstFloorPosts=%.3f",
      (unsigned long long)clockNano(),
      proxyState->tpRank,
      sub->peer,
      sub->channelId,
      sub->groupSize,
      ncclFuncToString((ncclFunc_t)args->coll),
      ncclFuncToString((ncclFunc_t)args->collAPI),
      ncclAlgoToString(args->algorithm),
      ncclProtoToString(args->protocol),
      ratioPct,
      observeMs,
      burstWindowMs,
      burstFloorPosts);
}

static inline void phase7BaselineLog(
    struct ncclProxyState* proxyState,
    struct ncclProxyArgs* args,
    struct ncclProxySubArgs* sub,
    uint64_t observedElapsedNs,
    double observedPosts,
    double baselineRatePerMs,
    double targetRatePerMs,
    double targetBurst) {
  if (ncclParamPhase7Log() == 0) return;
  INFO(NCCL_NET,
      "PHASE7 event=RATE_BASELINE tNs=%llu rank=%d peer=%d channel=%d groupSize=%d coll=%s collApi=%s algo=%s proto=%s observedElapsedNs=%llu observedPosts=%.3f baselineRatePerMs=%.6f targetRatePerMs=%.6f targetBurst=%.6f",
      (unsigned long long)clockNano(),
      proxyState->tpRank,
      sub->peer,
      sub->channelId,
      sub->groupSize,
      ncclFuncToString((ncclFunc_t)args->coll),
      ncclFuncToString((ncclFunc_t)args->collAPI),
      ncclAlgoToString(args->algorithm),
      ncclProtoToString(args->protocol),
      (unsigned long long)observedElapsedNs,
      observedPosts,
      baselineRatePerMs,
      targetRatePerMs,
      targetBurst);
}

static inline void phase7RateDecisionLog(
    struct ncclProxyState* proxyState,
    struct ncclProxyArgs* args,
    struct ncclProxySubArgs* sub,
    const char* event,
    uint64_t elapsedNs,
    int postCost,
    double ratioPct,
    double baselineRatePerMs,
    double targetRatePerMs,
    double targetBurst,
    double tokensBefore,
    double tokensAfter) {
  if (ncclParamPhase7Log() == 0) return;
  INFO(NCCL_NET,
      "PHASE7 event=%s tNs=%llu rank=%d peer=%d channel=%d groupSize=%d coll=%s collApi=%s algo=%s proto=%s elapsedNs=%llu postCost=%d ratioPct=%.3f baselineRatePerMs=%.6f targetRatePerMs=%.6f targetBurst=%.6f tokensBefore=%.6f tokensAfter=%.6f",
      event,
      (unsigned long long)clockNano(),
      proxyState->tpRank,
      sub->peer,
      sub->channelId,
      sub->groupSize,
      ncclFuncToString((ncclFunc_t)args->coll),
      ncclFuncToString((ncclFunc_t)args->collAPI),
      ncclAlgoToString(args->algorithm),
      ncclProtoToString(args->protocol),
      (unsigned long long)elapsedNs,
      postCost,
      ratioPct,
      baselineRatePerMs,
      targetRatePerMs,
      targetBurst,
      tokensBefore,
      tokensAfter);
}

static inline void phase9PostRateLog(
    struct ncclProxyState* proxyState,
    struct ncclProxyArgs* args,
    struct ncclProxySubArgs* leader,
    uint64_t nowNs,
    int postCost,
    uint64_t deltaNs,
    double instPostRatePerMs) {
  if (ncclParamPhase9Log() == 0) return;
  INFO(NCCL_NET,
      "PHASE9 event=RECVCOMM_POST_RATE tNs=%llu rank=%d peer=%d channel=%d groupSize=%d coll=%s collApi=%s algo=%s proto=%s postSeq=%llu postCost=%d deltaNs=%llu instPostRatePerMs=%.6f posted=%llu received=%llu done=%llu",
      (unsigned long long)nowNs,
      proxyState->tpRank,
      leader->peer,
      leader->channelId,
      leader->groupSize,
      ncclFuncToString((ncclFunc_t)args->coll),
      ncclFuncToString((ncclFunc_t)args->collAPI),
      ncclAlgoToString(args->algorithm),
      ncclProtoToString(args->protocol),
      (unsigned long long)leader->phase9PostSeq,
      postCost,
      (unsigned long long)deltaNs,
      instPostRatePerMs,
      (unsigned long long)leader->posted,
      (unsigned long long)leader->received,
      (unsigned long long)leader->done);
}

static inline uint64_t phase4Mix64(uint64_t x) {
  x ^= x >> 30;
  x *= 0xbf58476d1ce4e5b9ULL;
  x ^= x >> 27;
  x *= 0x94d049bb133111ebULL;
  x ^= x >> 31;
  return x;
}

static inline int phase4WindowEff(
    struct ncclProxyState* proxyState,
    struct ncclProxyArgs* args,
    struct ncclProxySubArgs* sub,
    double wRaw) {
  if (wRaw <= 0.0) return 0;
  int wFloor = (int)wRaw;
  double wFrac = wRaw - (double)wFloor;
  if (wFrac <= 0.0) return wFloor;

  uint64_t ticket = (uint64_t)sub->base + (uint64_t)sub->posted;
  uint64_t seed = ticket;
  seed ^= ((uint64_t)(proxyState->tpRank & 0xffff)) << 48;
  seed ^= ((uint64_t)(sub->peer & 0xffff)) << 32;
  seed ^= ((uint64_t)(sub->channelId & 0xffff)) << 16;
  seed ^= ((uint64_t)(args->coll & 0xff)) << 8;
  seed ^= (uint64_t)(args->protocol & 0xff);

  uint64_t mixed = phase4Mix64(seed);
  double unit = (double)(mixed >> 11) * (1.0 / 9007199254740992.0);
  return unit < wFrac ? (wFloor + 1) : wFloor;
}

struct phase2WindowDecision {
  int enabled;
  int rackSelf;
  int rackPeer;
  int rackKnown;
  int interRack;
  int penaltyTopo;
  int penaltyColl;
  int penaltyAlgo;
  int penaltyTotal;
  int wBase;
  int wMin;
  int wMax;
  int wRaw;
  int wEff;
};

static inline struct phase2WindowDecision phase2SelectWindow(
    struct ncclProxyState* proxyState,
    struct ncclProxyArgs* args,
    struct ncclProxySubArgs* sub) {
  (void)proxyState;
  (void)sub;
  struct phase2WindowDecision decision;
  memset(&decision, 0, sizeof(decision));
  decision.enabled = ncclParamPhase2B2Enable() != 0;
  decision.rackSelf = -1;
  decision.rackPeer = -1;
  decision.wBase = phase1WindowBaseDepth(args);
  decision.wMin = (args->collAPI == ncclFuncAlltoAll) ? 2 : 4;
  decision.wMax = decision.wBase;
  decision.wRaw = decision.wBase;
  decision.wEff = decision.wBase;

  if (phase1WindowCfg() > 0) return decision;

  if (args->collAPI == ncclFuncAlltoAll) decision.penaltyColl = 1;
  if (args->algorithm == NCCL_ALGO_TREE) decision.penaltyAlgo = 1;

  decision.penaltyTotal = decision.penaltyColl + decision.penaltyAlgo;
  decision.wRaw = decision.wBase - decision.penaltyTotal;
  decision.wEff = std::min(decision.wMax, std::max(decision.wMin, decision.wRaw));
  return decision;
}

struct phase3WindowDecision {
  int enabled;
  struct phase2WindowDecision semantic;
  int wSem;
  int wCur;
  int wFb;
  int wEff;
  int occPd;
  int occTr;
  int recvLag;
  uint64_t completionDelayNs;
  uint64_t delayBaseNs;
  uint64_t delayEwmaNs;
  int occRatioPct;
  int lagRatioPct;
  int delayRatioPct;
  int pressureScore;
  int hiCount;
  int loCount;
  int warmupCount;
  uint64_t ctrlStep;
  int oldW;
  int newW;
  const char* reason;
};

static inline int phase3Clamp(int lo, int hi, int value) {
  return std::min(hi, std::max(lo, value));
}

static inline int phase3RatioPct(uint64_t numerator, uint64_t denominator) {
  if (denominator == 0) return 0;
  return (int)((numerator * 100 + denominator - 1) / denominator);
}

static inline uint64_t phase3UpdateDelayEwma(uint64_t prev, uint64_t sample) {
  if (sample == 0) return prev;
  if (prev == 0) return sample;
  return (7 * prev + sample) / 8;
}

static inline struct phase3WindowDecision phase3SnapshotWindow(
    struct ncclProxySubArgs* sub,
    const struct phase2WindowDecision* semantic) {
  struct phase3WindowDecision decision;
  memset(&decision, 0, sizeof(decision));
  decision.enabled = ncclParamPhase3B3Enable() != 0;
  decision.semantic = *semantic;
  decision.wSem = semantic->wEff;
  if (sub->phase3CurrentW == 0) sub->phase3CurrentW = decision.wSem;
  decision.wCur = sub->phase3CurrentW;
  decision.wFb = decision.wCur;
  decision.wEff = phase3Clamp(semantic->wMin, semantic->wMax, std::min(decision.wSem, decision.wFb));
  decision.occPd = (int)(sub->posted - sub->done);
  decision.occTr = (int)(sub->transmitted - sub->done);
  decision.recvLag = (int)(sub->received - sub->transmitted);
  decision.completionDelayNs = sub->phase3LastDelayNs;
  decision.delayBaseNs = sub->phase3DelayBaseNs;
  decision.delayEwmaNs = sub->phase3DelayEwmaNs;
  decision.occRatioPct = phase3RatioPct((uint64_t)decision.occTr, (uint64_t)std::max(1, decision.wCur));
  decision.lagRatioPct = phase3RatioPct((uint64_t)decision.recvLag, (uint64_t)std::max(1, decision.wCur));
  decision.delayRatioPct = phase3RatioPct(decision.delayEwmaNs, std::max<uint64_t>(1, decision.delayBaseNs));
  decision.hiCount = sub->phase3HiCount;
  decision.loCount = sub->phase3LoCount;
  decision.warmupCount = sub->phase3WarmupCount;
  decision.ctrlStep = sub->phase3CtrlStep;
  decision.oldW = decision.wCur;
  decision.newW = decision.wEff;
  decision.reason = "init";
  return decision;
}

static inline struct phase3WindowDecision phase3UpdateController(
    struct ncclProxySubArgs* sub,
    const struct phase2WindowDecision* semantic) {
  struct phase3WindowDecision decision = phase3SnapshotWindow(sub, semantic);
  if (!decision.enabled || phase1WindowCfg() > 0) return decision;

  decision.ctrlStep = ++sub->phase3CtrlStep;
  decision.occPd = (int)(sub->posted - sub->done);
  decision.occTr = (int)(sub->transmitted - sub->done);
  decision.recvLag = (int)(sub->received - sub->transmitted);
  decision.completionDelayNs = sub->phase3LastDelayNs;
  decision.delayBaseNs = sub->phase3DelayBaseNs;
  decision.delayEwmaNs = sub->phase3DelayEwmaNs;
  decision.occRatioPct = phase3RatioPct((uint64_t)decision.occTr, (uint64_t)std::max(1, sub->phase3CurrentW));
  decision.lagRatioPct = phase3RatioPct((uint64_t)decision.recvLag, (uint64_t)std::max(1, sub->phase3CurrentW));
  decision.delayRatioPct = phase3RatioPct(decision.delayEwmaNs, std::max<uint64_t>(1, decision.delayBaseNs));

  decision.pressureScore = 0;
  if (decision.occRatioPct >= ncclParamPhase3OccRatioHighPct()) decision.pressureScore += 1;
  if (decision.lagRatioPct >= ncclParamPhase3LagRatioHighPct()) decision.pressureScore += 1;
  if (decision.delayRatioPct >= ncclParamPhase3DelayRatioHighPct()) decision.pressureScore += 1;

  decision.oldW = sub->phase3CurrentW;
  decision.newW = decision.oldW;
  decision.reason = "hold";

  if (sub->phase3WarmupCount < ncclParamPhase3WarmupIntervals()) {
    sub->phase3WarmupCount += 1;
    sub->phase3HiCount = 0;
    sub->phase3LoCount = 0;
    decision.reason = "warmup";
    decision.newW = decision.wSem;
  } else if (decision.pressureScore >= 2) {
    sub->phase3HiCount += 1;
    sub->phase3LoCount = 0;
    if (sub->phase3HiCount >= ncclParamPhase3HiIntervals()) {
      decision.newW = decision.oldW - 1;
      sub->phase3HiCount = 0;
      decision.reason = "shrink";
    }
  } else if (decision.pressureScore == 0) {
    sub->phase3LoCount += 1;
    sub->phase3HiCount = 0;
    if (sub->phase3LoCount >= ncclParamPhase3LoIntervals()) {
      decision.newW = decision.oldW + 1;
      sub->phase3LoCount = 0;
      decision.reason = "recover";
    }
  } else {
    sub->phase3HiCount = 0;
    sub->phase3LoCount = 0;
  }

  decision.wFb = decision.newW;
  decision.wEff = phase3Clamp(semantic->wMin, semantic->wMax, std::min(decision.wSem, decision.wFb));
  decision.newW = decision.wEff;
  sub->phase3CurrentW = decision.wEff;

  decision.wCur = sub->phase3CurrentW;
  decision.hiCount = sub->phase3HiCount;
  decision.loCount = sub->phase3LoCount;
  decision.warmupCount = sub->phase3WarmupCount;
  return decision;
}

static inline void phase1ProxyWindowCfgLog(
    struct ncclProxyState* proxyState,
    struct ncclProxyArgs* args,
    struct ncclProxySubArgs* sub,
    int shared,
    int wBase,
    int wCfg,
    int wEff) {
  if (ncclParamPhase0Log() == 0 || sub->phase1WindowCfgLogged) return;
  INFO(NCCL_NET,
      "PHASE1 event=PROXY_WINDOW_CFG tNs=%llu rank=%d peer=%d channel=%d coll=%s collApi=%s algo=%s proto=%s shared=%d nsubs=%d base=%llu nsteps=%d maxDepth=%d wBase=%d wCfg=%d wEff=%d",
      (unsigned long long)clockNano(),
      proxyState->tpRank,
      sub->peer,
      sub->channelId,
      ncclFuncToString((ncclFunc_t)args->coll),
      ncclFuncToString((ncclFunc_t)args->collAPI),
      ncclAlgoToString(args->algorithm),
      ncclProtoToString(args->protocol),
      shared,
      args->nsubs,
      (unsigned long long)sub->base,
      sub->nsteps,
      wBase,
      wBase,
      wCfg,
      wEff);
  sub->phase1WindowCfgLogged = 1;
}

static inline void phase1ProxyWstallLog(
    struct ncclProxyState* proxyState,
    struct ncclProxyArgs* args,
    struct ncclProxySubArgs* sub,
    const char* event,
    int slot,
    int wBase,
    int wCfg,
    int wEff,
    uint8_t* stallFlag) {
  (void)proxyState;
  (void)args;
  (void)sub;
  (void)event;
  (void)slot;
  (void)wBase;
  (void)wCfg;
  (void)wEff;
  (void)stallFlag;
  return;
}

static inline void appendix2RecvGroupLog(
    struct ncclProxyState* proxyState,
    struct ncclProxyArgs* args,
    struct ncclProxySubArgs* sub,
    int groupStart,
    int groupSize,
    int maxRecvs) {
  if (ncclParamAppendix2GroupLog() == 0) return;
  INFO(NCCL_NET,
      "APPENDIX2 event=RECV_GROUP_CFG tNs=%llu rank=%d peer=%d channel=%d groupStart=%d groupSize=%d maxRecvs=%d coll=%s collApi=%s algo=%s proto=%s base=%llu nsteps=%d",
      (unsigned long long)clockNano(),
      proxyState->tpRank,
      sub->peer,
      sub->channelId,
      groupStart,
      groupSize,
      maxRecvs,
      ncclFuncToString((ncclFunc_t)args->coll),
      ncclFuncToString((ncclFunc_t)args->collAPI),
      ncclAlgoToString(args->algorithm),
      ncclProtoToString(args->protocol),
      (unsigned long long)sub->base,
      sub->nsteps);
}

static inline void phase2ProxyWindowCfgLog(
    struct ncclProxyState* proxyState,
    struct ncclProxyArgs* args,
    struct ncclProxySubArgs* sub,
    const struct phase2WindowDecision* decision) {
  if (ncclParamPhase2Log() == 0 || sub->phase2WindowCfgLogged) return;
  INFO(NCCL_NET,
      "PHASE2 event=PROXY_B2_WINDOW_CFG tNs=%llu rank=%d peer=%d channel=%d coll=%s collApi=%s algo=%s proto=%s base=%llu nsteps=%d rackSelf=%d rackPeer=%d rackKnown=%d interRack=%d penTopo=%d penColl=%d penAlgo=%d penTotal=%d wBase=%d wMin=%d wMax=%d wRaw=%d wEff=%d",
      (unsigned long long)clockNano(),
      proxyState->tpRank,
      sub->peer,
      sub->channelId,
      ncclFuncToString((ncclFunc_t)args->coll),
      ncclFuncToString((ncclFunc_t)args->collAPI),
      ncclAlgoToString(args->algorithm),
      ncclProtoToString(args->protocol),
      (unsigned long long)sub->base,
      sub->nsteps,
      decision->rackSelf,
      decision->rackPeer,
      decision->rackKnown,
      decision->interRack,
      decision->penaltyTopo,
      decision->penaltyColl,
      decision->penaltyAlgo,
      decision->penaltyTotal,
      decision->wBase,
      decision->wMin,
      decision->wMax,
      decision->wRaw,
      decision->wEff);
  sub->phase2WindowCfgLogged = 1;
}

static inline void phase2ProxyRecvWstallLog(
    struct ncclProxyState* proxyState,
    struct ncclProxyArgs* args,
    struct ncclProxySubArgs* sub,
    int slot,
    const struct phase2WindowDecision* decision) {
  if (ncclParamAppendix2DisableWstallLog()) return;
  if (ncclParamPhase2Log() == 0 || sub->phase2RecvWstall) return;
  INFO(NCCL_NET,
      "PHASE2 event=PROXY_B2_RECV_WSTALL tNs=%llu rank=%d peer=%d channel=%d slot=%d coll=%s collApi=%s algo=%s proto=%s base=%llu posted=%llu received=%llu transmitted=%llu done=%llu nsteps=%d rackSelf=%d rackPeer=%d rackKnown=%d interRack=%d penTopo=%d penColl=%d penAlgo=%d penTotal=%d wBase=%d wMin=%d wMax=%d wRaw=%d wEff=%d occPd=%llu occTr=%llu",
      (unsigned long long)clockNano(),
      proxyState->tpRank,
      sub->peer,
      sub->channelId,
      slot,
      ncclFuncToString((ncclFunc_t)args->coll),
      ncclFuncToString((ncclFunc_t)args->collAPI),
      ncclAlgoToString(args->algorithm),
      ncclProtoToString(args->protocol),
      (unsigned long long)sub->base,
      (unsigned long long)sub->posted,
      (unsigned long long)sub->received,
      (unsigned long long)sub->transmitted,
      (unsigned long long)sub->done,
      sub->nsteps,
      decision->rackSelf,
      decision->rackPeer,
      decision->rackKnown,
      decision->interRack,
      decision->penaltyTopo,
      decision->penaltyColl,
      decision->penaltyAlgo,
      decision->penaltyTotal,
      decision->wBase,
      decision->wMin,
      decision->wMax,
      decision->wRaw,
      decision->wEff,
      (unsigned long long)(sub->posted - sub->done),
      (unsigned long long)(sub->transmitted - sub->done));
  sub->phase2RecvWstall = 1;
}

static inline void phase3ProxyWindowCfgLog(
    struct ncclProxyState* proxyState,
    struct ncclProxyArgs* args,
    struct ncclProxySubArgs* sub,
    const struct phase3WindowDecision* decision) {
  if (ncclParamPhase3Log() == 0) return;
  if (sub->phase3WindowCfgLogged && sub->phase3LastLoggedW == decision->wEff) return;
  INFO(NCCL_NET,
      "PHASE3 event=PROXY_B3_WINDOW_CFG tNs=%llu rank=%d peer=%d channel=%d coll=%s collApi=%s algo=%s proto=%s base=%llu nsteps=%d rackSelf=%d rackPeer=%d rackKnown=%d interRack=%d penTopo=%d penColl=%d penAlgo=%d penTotal=%d wBase=%d wMin=%d wMax=%d wRaw=%d wSem=%d wCur=%d wFb=%d wEff=%d ctrlStep=%llu",
      (unsigned long long)clockNano(),
      proxyState->tpRank,
      sub->peer,
      sub->channelId,
      ncclFuncToString((ncclFunc_t)args->coll),
      ncclFuncToString((ncclFunc_t)args->collAPI),
      ncclAlgoToString(args->algorithm),
      ncclProtoToString(args->protocol),
      (unsigned long long)sub->base,
      sub->nsteps,
      decision->semantic.rackSelf,
      decision->semantic.rackPeer,
      decision->semantic.rackKnown,
      decision->semantic.interRack,
      decision->semantic.penaltyTopo,
      decision->semantic.penaltyColl,
      decision->semantic.penaltyAlgo,
      decision->semantic.penaltyTotal,
      decision->semantic.wBase,
      decision->semantic.wMin,
      decision->semantic.wMax,
      decision->semantic.wRaw,
      decision->wSem,
      decision->oldW,
      decision->wFb,
      decision->wEff,
      (unsigned long long)decision->ctrlStep);
  sub->phase3WindowCfgLogged = 1;
  sub->phase3LastLoggedW = decision->wEff;
}

static inline void phase4ProxyWindowCfgLog(
    struct ncclProxyState* proxyState,
    struct ncclProxyArgs* args,
    struct ncclProxySubArgs* sub,
    int maxDepth,
    double wCfgRaw,
    int wEff) {
  if (ncclParamPhase4Log() == 0 || sub->phase4WindowCfgLogged) return;
  INFO(NCCL_NET,
      "PHASE4 event=PROXY_WINDOW_CFG tNs=%llu rank=%d peer=%d channel=%d coll=%s collApi=%s algo=%s proto=%s base=%llu nsteps=%d maxDepth=%d wCfgRaw=%.3f wEff=%d",
      (unsigned long long)clockNano(),
      proxyState->tpRank,
      sub->peer,
      sub->channelId,
      ncclFuncToString((ncclFunc_t)args->coll),
      ncclFuncToString((ncclFunc_t)args->collAPI),
      ncclAlgoToString(args->algorithm),
      ncclProtoToString(args->protocol),
      (unsigned long long)sub->base,
      sub->nsteps,
      maxDepth,
      wCfgRaw,
      wEff);
  sub->phase4WindowCfgLogged = 1;
}

static inline void phase4ProxyMaxDepthLog(
    struct ncclProxyState* proxyState,
    struct ncclProxyArgs* args,
    struct ncclProxySubArgs* sub,
    int maxDepth,
    int phase4Enabled,
    double wCfgRaw) {
  if (args->phase4MaxDepthLogged) return;
  if (ncclParamPhase0Log() == 0 && ncclParamPhase4Log() == 0) return;
  INFO(NCCL_NET,
      "PHASE4 event=PROXY_MAX_DEPTH tNs=%llu rank=%d peer=%d channel=%d coll=%s collApi=%s algo=%s proto=%s base=%llu nsteps=%d maxDepth=%d phase4Enable=%d wCfgRaw=%.3f",
      (unsigned long long)clockNano(),
      proxyState->tpRank,
      sub->peer,
      sub->channelId,
      ncclFuncToString((ncclFunc_t)args->coll),
      ncclFuncToString((ncclFunc_t)args->collAPI),
      ncclAlgoToString(args->algorithm),
      ncclProtoToString(args->protocol),
      (unsigned long long)sub->base,
      sub->nsteps,
      maxDepth,
      phase4Enabled,
      wCfgRaw);
  args->phase4MaxDepthLogged = 1;
}

static inline void phase4ProxyWstallLog(
    struct ncclProxyState* proxyState,
    struct ncclProxyArgs* args,
    struct ncclProxySubArgs* sub,
    const char* event,
    int slot,
    double wCfgRaw,
    int wEff,
    uint8_t* stallFlag) {
  if (ncclParamAppendix2DisableWstallLog()) return;
  if (ncclParamPhase4Log() == 0 || *stallFlag) return;
  INFO(NCCL_NET,
      "PHASE4 event=%s tNs=%llu rank=%d peer=%d channel=%d slot=%d coll=%s collApi=%s algo=%s proto=%s base=%llu posted=%llu received=%llu transmitted=%llu done=%llu nsteps=%d wCfgRaw=%.3f wEff=%d occPr=%llu occPd=%llu",
      event,
      (unsigned long long)clockNano(),
      proxyState->tpRank,
      sub->peer,
      sub->channelId,
      slot,
      ncclFuncToString((ncclFunc_t)args->coll),
      ncclFuncToString((ncclFunc_t)args->collAPI),
      ncclAlgoToString(args->algorithm),
      ncclProtoToString(args->protocol),
      (unsigned long long)sub->base,
      (unsigned long long)sub->posted,
      (unsigned long long)sub->received,
      (unsigned long long)sub->transmitted,
      (unsigned long long)sub->done,
      sub->nsteps,
      wCfgRaw,
      wEff,
      (unsigned long long)(sub->posted - sub->received),
      (unsigned long long)(sub->posted - sub->done));
  *stallFlag = 1;
}

static inline void phase3ProxyPressureLog(
    struct ncclProxyState* proxyState,
    struct ncclProxyArgs* args,
    struct ncclProxySubArgs* sub,
    const struct phase3WindowDecision* decision) {
  if (ncclParamPhase3Log() == 0) return;
  INFO(NCCL_NET,
      "PHASE3 event=PROXY_B3_PRESSURE tNs=%llu rank=%d peer=%d channel=%d coll=%s collApi=%s algo=%s proto=%s ctrlStep=%llu occPd=%d occTr=%d recvLag=%d completionDelayNs=%llu delayBaseNs=%llu delayEwmaNs=%llu occRatioPct=%d lagRatioPct=%d delayRatioPct=%d pressureScore=%d hiCount=%d loCount=%d wSem=%d wCur=%d wEff=%d",
      (unsigned long long)clockNano(),
      proxyState->tpRank,
      sub->peer,
      sub->channelId,
      ncclFuncToString((ncclFunc_t)args->coll),
      ncclFuncToString((ncclFunc_t)args->collAPI),
      ncclAlgoToString(args->algorithm),
      ncclProtoToString(args->protocol),
      (unsigned long long)decision->ctrlStep,
      decision->occPd,
      decision->occTr,
      decision->recvLag,
      (unsigned long long)decision->completionDelayNs,
      (unsigned long long)decision->delayBaseNs,
      (unsigned long long)decision->delayEwmaNs,
      decision->occRatioPct,
      decision->lagRatioPct,
      decision->delayRatioPct,
      decision->pressureScore,
      decision->hiCount,
      decision->loCount,
      decision->wSem,
      decision->oldW,
      decision->wEff);
}

static inline void phase3ProxyDecisionLog(
    struct ncclProxyState* proxyState,
    struct ncclProxyArgs* args,
    struct ncclProxySubArgs* sub,
    const struct phase3WindowDecision* decision) {
  if (ncclParamPhase3Log() == 0) return;
  INFO(NCCL_NET,
      "PHASE3 event=PROXY_B3_DECISION tNs=%llu rank=%d peer=%d channel=%d coll=%s collApi=%s algo=%s proto=%s ctrlStep=%llu reason=%s oldW=%d newW=%d wSem=%d wFb=%d wEff=%d pressureScore=%d hiCount=%d loCount=%d",
      (unsigned long long)clockNano(),
      proxyState->tpRank,
      sub->peer,
      sub->channelId,
      ncclFuncToString((ncclFunc_t)args->coll),
      ncclFuncToString((ncclFunc_t)args->collAPI),
      ncclAlgoToString(args->algorithm),
      ncclProtoToString(args->protocol),
      (unsigned long long)decision->ctrlStep,
      decision->reason,
      decision->oldW,
      decision->newW,
      decision->wSem,
      decision->wFb,
      decision->wEff,
      decision->pressureScore,
      decision->hiCount,
      decision->loCount);
}

static inline void phase3ProxyRecvWstallLog(
    struct ncclProxyState* proxyState,
    struct ncclProxyArgs* args,
    struct ncclProxySubArgs* sub,
    int slot,
    const struct phase3WindowDecision* decision) {
  if (ncclParamAppendix2DisableWstallLog()) return;
  if (ncclParamPhase3Log() == 0 || sub->phase3RecvWstall) return;
  INFO(NCCL_NET,
      "PHASE3 event=PROXY_B3_RECV_WSTALL tNs=%llu rank=%d peer=%d channel=%d slot=%d coll=%s collApi=%s algo=%s proto=%s base=%llu posted=%llu received=%llu transmitted=%llu done=%llu nsteps=%d ctrlStep=%llu wSem=%d wEff=%d occPd=%d occTr=%d recvLag=%d pressureScore=%d",
      (unsigned long long)clockNano(),
      proxyState->tpRank,
      sub->peer,
      sub->channelId,
      slot,
      ncclFuncToString((ncclFunc_t)args->coll),
      ncclFuncToString((ncclFunc_t)args->collAPI),
      ncclAlgoToString(args->algorithm),
      ncclProtoToString(args->protocol),
      (unsigned long long)sub->base,
      (unsigned long long)sub->posted,
      (unsigned long long)sub->received,
      (unsigned long long)sub->transmitted,
      (unsigned long long)sub->done,
      sub->nsteps,
      (unsigned long long)decision->ctrlStep,
      decision->wSem,
      decision->wEff,
      decision->occPd,
      decision->occTr,
      decision->recvLag,
      decision->pressureScore);
  sub->phase3RecvWstall = 1;
}

static ncclResult_t sharedNetBuffersInit(struct ncclProxyState* proxyState, int cuda, int tpLocalRank, int type, int sameProcess,
    int nChannels, char** gpuPtr, char** cpuPtr, int* size, ncclIpcDesc *ipcDesc) {
  if (cuda == 0 && sameProcess == 0) {
      WARN("PXN should not use host buffers for data");
      return ncclInternalError;
  }
  struct ncclProxyProgressState* progressState = &proxyState->progressState;
  if (progressState->localPeers == NULL) {
    NCCLCHECK(ncclCalloc(&progressState->localPeers, proxyState->tpLocalnRanks));
  }
  struct ncclProxyPeer** localPeers = progressState->localPeers;
  if (localPeers[tpLocalRank] == NULL) {
    NCCLCHECK(ncclCalloc(localPeers + tpLocalRank, 1));
  }
  struct ncclProxyPeer* peer = localPeers[tpLocalRank];
  struct ncclProxySharedP2p* state = type == 0 ? &peer->send : &peer->recv;
  state->refcount++;
  if (state->size == 0) {
    state->size = nChannels * NCCL_SHARED_STEPS * proxyState->p2pChunkSize;
  }

  if (size) *size = state->size;

  if (cuda && state->cudaBuff == NULL) {
    if (sameProcess == 0 || ncclCuMemEnable()) {
      NCCLCHECK(ncclP2pAllocateShareableBuffer(state->size, 0, &state->ipcDesc, (void**)&state->cudaBuff));
    } else {
      NCCLCHECK(ncclCudaCalloc(&state->cudaBuff, state->size, proxyState->memManager));
    }
  }
  if (!cuda && state->hostBuff == NULL) {
    NCCLCHECK(ncclCudaHostCalloc(&state->hostBuff, state->size));
  }
  if (cpuPtr) *cpuPtr = cuda ? state->cudaBuff : state->hostBuff;
  if (gpuPtr) *gpuPtr = (cpuPtr && sameProcess) ? *cpuPtr : NULL;
  if (ipcDesc) memcpy(ipcDesc, &state->ipcDesc, sizeof(state->ipcDesc));
  return ncclSuccess;
}

static ncclResult_t sharedBuffersGet(struct ncclProxyState* proxyState, int channel, int slot, int* offset, size_t* size) {
  // Use different pools for different channels and also separate send/recv.
  int globalSlot = (channel*NCCL_SHARED_STEPS)+slot;
  *offset = proxyState->p2pChunkSize * globalSlot;
  if (size) *size = proxyState->p2pChunkSize;
  return ncclSuccess;
}

static ncclResult_t sharedNetBuffersDestroy(struct ncclProxyState* proxyState, int tpLocalRank, int type, struct ncclProxyConnection* connection) {
  if (proxyState->progressState.localPeers == NULL) NCCLCHECK(ncclInternalError);
  struct ncclProxyPeer* peer = proxyState->progressState.localPeers[tpLocalRank];
  if (peer == NULL) NCCLCHECK(ncclInternalError);
  struct ncclProxySharedP2p* state = type == 0 ? &peer->send : &peer->recv;
  if (state->size == 0) NCCLCHECK(ncclInternalError);
  if (ncclAtomicRefCountDecrement(&state->refcount) == 0) {
    if (state->cudaBuff) {
      if (!connection->sameProcess || ncclCuMemEnable()) {
        NCCLCHECK(ncclP2pFreeShareableBuffer(&state->ipcDesc));
      }
      NCCLCHECK(ncclCudaFree(state->cudaBuff, proxyState->memManager));
    }
    if (state->hostBuff) NCCLCHECK(ncclCudaHostFree(state->hostBuff));
  }

  if (peer->send.refcount || peer->recv.refcount) return ncclSuccess;

  free(peer);
  proxyState->progressState.localPeers[tpLocalRank] = NULL;
  for (int r = 0; r < proxyState->tpLocalnRanks; r++) {
    if (proxyState->progressState.localPeers[r]) return ncclSuccess;
  }
  // All peers are freed, free array
  free(proxyState->progressState.localPeers);
  proxyState->progressState.localPeers = NULL;
  return ncclSuccess;
}

static ncclResult_t proxySharedInit(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, int nChannels) {
  NCCLCHECK(sharedNetBuffersInit(proxyState, 1, connection->tpLocalRank, 0, connection->sameProcess, nChannels, NULL, NULL, NULL, NULL));
  return ncclSuccess;
}

static ncclResult_t sendProxySetup(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, void* reqBuff, int reqSize, void* respBuff, int respSize, int* done) {
  struct setupReq* req = (struct setupReq*) reqBuff;
  if (reqSize != sizeof(struct setupReq)) return ncclInternalError;

  struct sendNetResources* resources;
  NCCLCHECK(ncclCalloc(&resources, 1));
  connection->transportResources = resources;

  resources->tpRank = req->tpRank;
  resources->tpLocalRank = req->tpLocalRank;
  resources->tpRemoteRank = req->tpRemoteRank;
  resources->netDev = req->netDev;
  resources->shared = connection->shared = req->shared;
  resources->useGdr = req->useGdr;
  resources->channelId = req->channelId;
  resources->connIndex = req->connIndex;
  ncclNetProperties_t props;
  NCCLCHECK(proxyState->ncclNet->getProperties(req->netDev, &props));
  /* DMA-BUF support */
  resources->useDmaBuf = resources->useGdr && proxyState->dmaBufSupport && (props.ptrSupport & NCCL_PTR_DMABUF);
  resources->maxRecvs = props.maxRecvs;
  resources->netDeviceVersion = props.netDeviceVersion;
  resources->netDeviceType = props.netDeviceType;

  /* point-to-point size limits*/
  resources->maxP2pBytes = props.maxP2pBytes;
  if((resources->maxP2pBytes <= 0) || (resources->maxP2pBytes > NCCL_MAX_NET_SIZE_BYTES)) {
    WARN("sendProxySetup: net plugin returned invalid value for maxP2pBytes %ld \
      [allowed range: %ld - %ld] \n", resources->maxP2pBytes, 0L, NCCL_MAX_NET_SIZE_BYTES);
    return ncclInternalError;
  }

  // We don't return any data
  if (respSize != 0) return ncclInternalError;
  *done = 1;
  return ncclSuccess;
}

static ncclResult_t recvProxySetup(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, void* reqBuff, int reqSize, void* respBuff, int respSize, int* done) {
  struct setupReq* req = (struct setupReq*) reqBuff;
  if (reqSize != sizeof(struct setupReq)) return ncclInternalError;

  struct recvNetResources* resources;
  NCCLCHECK(ncclCalloc(&resources, 1));
  connection->transportResources = resources;

  resources->tpRank = req->tpRank;
  resources->tpLocalRank = req->tpLocalRank;
  resources->tpRemoteRank = req->tpRemoteRank;
  resources->netDev = req->netDev;
  resources->shared = connection->shared = req->shared;
  resources->useGdr = req->useGdr;
  resources->needFlush = req->needFlush;
  resources->channelId = req->channelId;
  resources->connIndex = req->connIndex;
  ncclNetProperties_t props;
  NCCLCHECK(proxyState->ncclNet->getProperties(req->netDev, &props));
  /* DMA-BUF support */
  resources->useDmaBuf = resources->useGdr && proxyState->dmaBufSupport && (props.ptrSupport & NCCL_PTR_DMABUF);
  resources->maxRecvs = props.maxRecvs;
  resources->netDeviceVersion = props.netDeviceVersion;
  resources->netDeviceType = props.netDeviceType;
  /* point-to-point size limits*/
  resources->maxP2pBytes = props.maxP2pBytes;
  if((resources->maxP2pBytes <= 0) || (resources->maxP2pBytes > NCCL_MAX_NET_SIZE_BYTES)) {
    WARN("recvProxySetup: net plugin returned invalid value for maxP2pBytes %ld \
      [allowed range: %ld - %ld] \n", resources->maxP2pBytes, 0L, NCCL_MAX_NET_SIZE_BYTES);
    return ncclInternalError;
  }

  if (respSize != sizeof(ncclNetHandle_t)) return ncclInternalError;
  NCCLCHECK(proxyState->ncclNet->listen(proxyState->netContext, req->netDev, respBuff, &resources->netListenComm));
  *done = 1;

  return ncclSuccess;
}

// This function embeds plugin-specific rules given the current versions
static ncclResult_t ncclNetGetDeviceHandle(ncclNetDeviceType type, int version, bool isRecv, ncclNetDeviceHandle_t** handle) {
  bool needsDeviceHandle  = false;

  if (type == NCCL_NET_DEVICE_UNPACK) {
    if (version == NCCL_NET_DEVICE_UNPACK_VERSION && isRecv) {
      needsDeviceHandle  = true;
    }
  }

  // Don't re-alloc netDeviceHandles
  if (needsDeviceHandle && (*handle == NULL)) {
    NCCLCHECK(ncclCalloc(handle, 1));
    (*handle)->netDeviceType = type;
    (*handle)->netDeviceVersion = version;
  } else if (!needsDeviceHandle) {
    *handle = NULL;
  }

  return ncclSuccess;
}

static ncclResult_t sendProxyConnect(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, void* reqBuff, int reqSize, void* respBuff, int respSize, int* done) {
  struct sendNetResources* resources = (struct sendNetResources*)(connection->transportResources);
  if (reqSize != sizeof(netSendConnectArgs)) return ncclInternalError;
  ncclResult_t ret = ncclSuccess;
  netSendConnectArgs* req = (netSendConnectArgs*) reqBuff;

  setNetAttrs(proxyState, &req->netAttr);

  NCCLCHECK(ncclNetGetDeviceHandle(resources->netDeviceType, resources->netDeviceVersion, false /*isRecv*/, &resources->netDeviceHandle));
  if (resources->shared) {
    // Shared buffers
    struct ncclProxyProgressState* progressState = &proxyState->progressState;
    if (progressState->localPeers == NULL) {
      NCCLCHECK(ncclCalloc(&progressState->localPeers, proxyState->tpLocalnRanks));
    }
    struct ncclProxyPeer** localPeers = progressState->localPeers;
    if (localPeers[resources->tpLocalRank] == NULL) {
      NCCLCHECK(ncclCalloc(localPeers + resources->tpLocalRank, 1));
    }
    connection->proxyAppendPtr = localPeers[resources->tpLocalRank]->send.proxyAppend + resources->channelId;

    if (resources->maxRecvs > 1 && ncclParamNetSharedComms()) {
      // Connect or reuse connection for a netdev/remote rank.
      if (progressState->netComms[resources->netDev] == NULL) {
        NCCLCHECK(ncclCalloc(progressState->netComms + resources->netDev, proxyState->tpnRanks));
      }
      struct ncclSharedNetComms* comms = progressState->netComms[resources->netDev] + resources->tpRemoteRank;
      // let only one localrank connect to a tpRemoteRank to avoid duplicate connections
      if (comms->activeConnect[resources->channelId] == 0)
        comms->activeConnect[resources->channelId] = (resources->tpLocalRank + 1);
      if (comms->sendComm[resources->channelId] == NULL
          && comms->activeConnect[resources->channelId] == (resources->tpLocalRank + 1)) {
        ret = proxyState->ncclNet->connect(proxyState->netContext, resources->netDev, req->handle,
            comms->sendComm + resources->channelId, &resources->netDeviceHandle);
      }
      resources->netSendComm = comms->sendComm[resources->channelId];
      if (comms->sendComm[resources->channelId]) comms->sendRefCount[resources->channelId]++;
    } else {
      ret = proxyState->ncclNet->connect(proxyState->netContext, resources->netDev, req->handle, &resources->netSendComm, &resources->netDeviceHandle);
    }
  } else {
    // Connect to remote peer
    ret = proxyState->ncclNet->connect(proxyState->netContext, resources->netDev, req->handle, &resources->netSendComm, &resources->netDeviceHandle);
    connection->proxyAppendPtr = &connection->proxyAppend;
  }

  if (ret != ncclSuccess) {
    if (resources->netSendComm) proxyState->ncclNet->closeSend(resources->netSendComm);
    NCCLCHECK(ret);
  }
  if (resources->netSendComm == NULL) {
    *done = 0;
    return ncclInProgress;
  }
  printNetAttrs(&req->netAttr, "send connect");
  *done = 1;

  if (resources->netDeviceHandle) {
    connection->netDeviceHandle = resources->netDeviceHandle;
    connection->needsProxyProgress = connection->netDeviceHandle->needsProxyProgress;
  } else {
    connection->needsProxyProgress = 1;
  }

  // Create structures
  struct connectMap* map = &resources->map;
  map->sameProcess = connection->sameProcess;
  map->shared = resources->shared;
  CUDACHECK(cudaGetDevice(&map->cudaDev));

  if (resources->shared == 0) { // Only allocate dedicated buffers for ring/tree, not for p2p
    for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
      NCCL_NET_MAP_ADD_POINTER(map, 0, p!= NCCL_PROTO_LL && resources->useGdr ? 1 : 0, proxyState->buffSizes[p], buffs[p]);
      resources->buffSizes[p] = proxyState->buffSizes[p];
    }
  } else {
    // Get shared buffers
    int bank = resources->useGdr ? NCCL_NET_MAP_SHARED_DEVMEM : NCCL_NET_MAP_SHARED_HOSTMEM;
    struct connectMapMem* mapMem = map->mems+bank;
    NCCLCHECK(sharedNetBuffersInit(
          proxyState, resources->useGdr, resources->tpLocalRank, 0, map->sameProcess, proxyState->p2pnChannels,
          &mapMem->gpuPtr, &mapMem->cpuPtr, &mapMem->size, &mapMem->ipcDesc));
    resources->buffSizes[NCCL_PROTO_SIMPLE] = mapMem->size;

    if (proxyState->allocP2pNetLLBuffers) {
      NCCL_NET_MAP_ADD_POINTER(map, 0, 0 /*p == NCCL_PROTO_LL*/, proxyState->buffSizes[NCCL_PROTO_LL], buffs[NCCL_PROTO_LL]);
      resources->buffSizes[NCCL_PROTO_LL] = proxyState->buffSizes[NCCL_PROTO_LL];
    }

    NCCL_NET_MAP_ADD_POINTER(map, 1, resources->useGdr ? 1 : 0, mapMem->size, buffs[NCCL_PROTO_SIMPLE]);
  }

  NCCL_NET_MAP_ADD_POINTER(map, 0, 0, sizeof(struct ncclSendMem), sendMem);
  NCCL_NET_MAP_ADD_POINTER(map, 0, 0, sizeof(struct ncclRecvMem), recvMem);

  if (map->mems[NCCL_NET_MAP_DEVMEM].size) {
    if (resources->shared == 0) {
      if (!map->sameProcess || ncclCuMemEnable()) {
        ALIGN_SIZE(map->mems[NCCL_NET_MAP_DEVMEM].size, CUDA_IPC_MIN);
        NCCLCHECK(ncclP2pAllocateShareableBuffer(map->mems[NCCL_NET_MAP_DEVMEM].size, 0, &map->mems[NCCL_NET_MAP_DEVMEM].ipcDesc,
                                                 (void**)&map->mems[NCCL_NET_MAP_DEVMEM].gpuPtr));
      } else {
        NCCLCHECK(ncclCudaCalloc(&map->mems[NCCL_NET_MAP_DEVMEM].gpuPtr, map->mems[NCCL_NET_MAP_DEVMEM].size, proxyState->memManager));
      }
      map->mems[NCCL_NET_MAP_DEVMEM].cpuPtr = map->mems[NCCL_NET_MAP_DEVMEM].gpuPtr;
    }
  }
  if (map->sameProcess) {
    NCCLCHECK(ncclCudaHostCalloc(&map->mems[NCCL_NET_MAP_HOSTMEM].cpuPtr, map->mems[NCCL_NET_MAP_HOSTMEM].size));
    map->mems[NCCL_NET_MAP_HOSTMEM].gpuPtr = map->mems[NCCL_NET_MAP_HOSTMEM].cpuPtr;
  } else {
    NCCLCHECK(netCreateShm(proxyState, map->mems+NCCL_NET_MAP_HOSTMEM));
    void* sendMem = (void*)NCCL_NET_MAP_GET_POINTER(map, cpu, sendMem);
    void* recvMem = (void*)NCCL_NET_MAP_GET_POINTER(map, cpu, recvMem);
    memset(sendMem, 0, sizeof(struct ncclSendMem));
    memset(recvMem, 0, sizeof(struct ncclRecvMem));
  }
  if (ncclGdrCopy && map->sameProcess && ncclParamGdrCopySyncEnable()) {
    uint64_t *cpuPtr, *gpuPtr;
    NCCLCHECK(ncclGdrCudaCalloc(&cpuPtr, &gpuPtr, 1, &resources->gdrDesc, proxyState->memManager));

    resources->gdcSync = cpuPtr;
    struct connectMapMem* gdcMem = map->mems+NCCL_NET_MAP_GDCMEM;
    gdcMem->cpuPtr = (char*)cpuPtr;
    gdcMem->gpuPtr = (char*)gpuPtr;
    gdcMem->size = sizeof(uint64_t); // sendMem->head
  }

  resources->sendMem = (struct ncclSendMem*) NCCL_NET_MAP_GET_POINTER(map, cpu, sendMem);
  resources->recvMem = (struct ncclRecvMem*) NCCL_NET_MAP_GET_POINTER(map, cpu, recvMem);

  // Don't give credits yet in shared mode.
  (resources->gdcSync ? *resources->gdcSync : resources->sendMem->head) =
    (map->shared ? -NCCL_STEPS : 0);
  for (int i=0; i<NCCL_STEPS; i++) resources->recvMem->connFifo[i].size = -1;

  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
    resources->buffers[p] = NCCL_NET_MAP_GET_POINTER(map, cpu, buffs[p]);
    if (resources->buffers[p]) {
#if CUDA_VERSION >= 11070
      /* DMA-BUF support */
      int type = NCCL_NET_MAP_DEV_MEM(map, buffs[p]) ? NCCL_PTR_CUDA : NCCL_PTR_HOST;
      if (type == NCCL_PTR_CUDA && resources->useDmaBuf) {
        int dmabuf_fd;
        CUCHECK(cuMemGetHandleForAddressRange((void *)&dmabuf_fd, (CUdeviceptr)resources->buffers[p], resources->buffSizes[p], CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, getHandleForAddressRangeFlags(resources->useGdr)));
        NCCLCHECK(proxyState->ncclNet->regMrDmaBuf(resources->netSendComm, resources->buffers[p], resources->buffSizes[p], type, 0ULL, dmabuf_fd, &resources->mhandles[p]));
        (void)close(dmabuf_fd);
      } else // FALL-THROUGH to nv_peermem GDR path
#endif
      {
        NCCLCHECK(proxyState->ncclNet->regMr(resources->netSendComm, resources->buffers[p], resources->buffSizes[p], NCCL_NET_MAP_DEV_MEM(map, buffs[p]) ? NCCL_PTR_CUDA : NCCL_PTR_HOST, &resources->mhandles[p]));
      }

      // Copy the mhandle dptr, if implemented
      if (resources->netDeviceHandle && proxyState->ncclNet->getDeviceMr)
        NCCLCHECK(proxyState->ncclNet->getDeviceMr(resources->netSendComm, resources->mhandles[p], &connection->mhandles[p]));
    }
  }

  //NCCLCHECK(netDumpMap(map));
  if (respSize != sizeof(struct connectMap)) return ncclInternalError;
  memcpy(respBuff, map, sizeof(struct connectMap));
  return ncclSuccess;
}

static ncclResult_t recvProxyConnect(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, void* reqBuff, int reqSize, void* respBuff, int respSize, int* done) {
  if (reqSize != sizeof(netRecvConnectArgs)) return ncclInternalError;
  struct recvNetResources* resources = (struct recvNetResources*)(connection->transportResources);
  netRecvConnectArgs* req = (netRecvConnectArgs*) reqBuff;
  resources->tpRemoteProxyRank = req->proxyRank;
  ncclResult_t ret = ncclSuccess;

  setNetAttrs(proxyState, &req->netAttr);

  NCCLCHECK(ncclNetGetDeviceHandle(resources->netDeviceType, resources->netDeviceVersion, true /*isRecv*/, &resources->netDeviceHandle));
  // Finish connection establishment from remote peer
  if (resources->shared) {
    // Shared buffers
    struct ncclProxyProgressState* progressState = &proxyState->progressState;
    if (progressState->localPeers == NULL) {
      NCCLCHECK(ncclCalloc(&progressState->localPeers, proxyState->tpLocalnRanks));
    }
    struct ncclProxyPeer** localPeers = progressState->localPeers;
    if (localPeers[resources->tpLocalRank] == NULL) {
      NCCLCHECK(ncclCalloc(localPeers + resources->tpLocalRank, 1));
    }
    connection->proxyAppendPtr = localPeers[resources->tpLocalRank]->recv.proxyAppend + resources->channelId;

    if (resources->maxRecvs > 1 && ncclParamNetSharedComms()) {
      // Connect or reuse connection for a netdev/remote rank.
      if (progressState->netComms[resources->netDev] == NULL) {
        NCCLCHECK(ncclCalloc(progressState->netComms + resources->netDev, proxyState->tpnRanks));
      }
      struct ncclSharedNetComms* comms = progressState->netComms[resources->netDev] + resources->tpRemoteProxyRank;
      // reuse handle to for netdev/remote rank to avoid duplicate connections
      if (comms->activeAccept[resources->channelId] == 0)
        comms->activeAccept[resources->channelId] = (resources->tpLocalRank + 1);
      //try connecting while comm is null
      if (comms->recvComm[resources->channelId] == NULL
         && comms->activeAccept[resources->channelId] == (resources->tpLocalRank + 1)) {
        ret = proxyState->ncclNet->accept(resources->netListenComm,
            comms->recvComm+resources->channelId, &resources->netDeviceHandle);
      }
      resources->netRecvComm = comms->recvComm[resources->channelId];
      if (comms->recvComm[resources->channelId]) comms->recvRefCount[resources->channelId]++;
    } else {
      ret = proxyState->ncclNet->accept(resources->netListenComm, &resources->netRecvComm, &resources->netDeviceHandle);
    }
  } else {
    // Connect to remote peer
    ret = proxyState->ncclNet->accept(resources->netListenComm, &resources->netRecvComm, &resources->netDeviceHandle);
    connection->proxyAppendPtr = &connection->proxyAppend;
  }

  NCCLCHECK(ret);
  if (resources->netRecvComm == NULL) {
    *done = 0;
    return ncclInProgress;
  }
  printNetAttrs(&req->netAttr, "recv connect");
  *done = 1;

  if (resources->netDeviceHandle) {
    connection->netDeviceHandle = resources->netDeviceHandle;
    connection->needsProxyProgress = connection->netDeviceHandle->needsProxyProgress;
  } else {
    connection->needsProxyProgress = 1;
  }

  NCCLCHECK(proxyState->ncclNet->closeListen(resources->netListenComm));

  // Create structures
  struct connectMap* map = &resources->map;
  map->sameProcess = connection->sameProcess;
  if (map->sameProcess == 0) return ncclInternalError; // We don't support remote proxy for recv
  map->shared = resources->shared;

  if (resources->shared == 0) { // Only allocate dedicated buffers for ring/tree, not for p2p
    for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
      NCCL_NET_MAP_ADD_POINTER(map, 0, resources->useGdr ? 1 : 0, proxyState->buffSizes[p], buffs[p]);
      resources->buffSizes[p] = proxyState->buffSizes[p];
    }
  } else {
    // Get shared buffers
    int bank = resources->useGdr ? NCCL_NET_MAP_SHARED_DEVMEM : NCCL_NET_MAP_SHARED_HOSTMEM;
    struct connectMapMem* mapMem = map->mems+bank;
    NCCLCHECK(sharedNetBuffersInit(
          proxyState, resources->useGdr, resources->tpLocalRank, 1, 1, proxyState->p2pnChannels,
          &mapMem->gpuPtr, &mapMem->cpuPtr, &mapMem->size, NULL));
    resources->buffSizes[NCCL_PROTO_SIMPLE] = mapMem->size;
    NCCL_NET_MAP_ADD_POINTER(map, 1, resources->useGdr ? 1 : 0, mapMem->size, buffs[NCCL_PROTO_SIMPLE]);
  }

  NCCL_NET_MAP_ADD_POINTER(map, 0, 0, sizeof(struct ncclSendMem), sendMem);
  NCCL_NET_MAP_ADD_POINTER(map, 0, 0, sizeof(struct ncclRecvMem), recvMem);

  if (proxyState->allocP2pNetLLBuffers) {
    NCCL_NET_MAP_ADD_POINTER(map, 0, 0 /*devMem*/, proxyState->buffSizes[NCCL_PROTO_LL], buffs[NCCL_PROTO_LL]);
    resources->buffSizes[NCCL_PROTO_LL] = proxyState->buffSizes[NCCL_PROTO_LL];
  }

  if (map->mems[NCCL_NET_MAP_DEVMEM].size) {
    if (resources->shared == 0) {
      if (ncclCuMemEnable()) {
        NCCLCHECK(ncclP2pAllocateShareableBuffer(map->mems[NCCL_NET_MAP_DEVMEM].size, 0, &map->mems[NCCL_NET_MAP_DEVMEM].ipcDesc,
                                                 (void**)&map->mems[NCCL_NET_MAP_DEVMEM].gpuPtr));
      } else {
        NCCLCHECK(ncclCudaCalloc(&map->mems[NCCL_NET_MAP_DEVMEM].gpuPtr, map->mems[NCCL_NET_MAP_DEVMEM].size, proxyState->memManager));
      }
      map->mems[NCCL_NET_MAP_DEVMEM].cpuPtr = map->mems[NCCL_NET_MAP_DEVMEM].gpuPtr;
    }
  }
  NCCLCHECK(ncclCudaHostCalloc(&map->mems[NCCL_NET_MAP_HOSTMEM].cpuPtr, map->mems[NCCL_NET_MAP_HOSTMEM].size));
  map->mems[NCCL_NET_MAP_HOSTMEM].gpuPtr = map->mems[NCCL_NET_MAP_HOSTMEM].cpuPtr;
  if (ncclGdrCopy && map->sameProcess) {
    uint64_t *cpuPtr, *gpuPtr;
    NCCLCHECK(ncclGdrCudaCalloc(&cpuPtr, &gpuPtr, 2, &resources->gdrDesc, proxyState->memManager));

    if (ncclParamGdrCopySyncEnable()) {
      resources->gdcSync = cpuPtr;
      struct connectMapMem* gdcMem = map->mems+NCCL_NET_MAP_GDCMEM;
      gdcMem->cpuPtr = (char*)cpuPtr;
      gdcMem->gpuPtr = (char*)gpuPtr;
      gdcMem->size = sizeof(uint64_t);
    }
    if (ncclParamGdrCopyFlushEnable()) resources->gdcFlush = cpuPtr + 1;
  }

  resources->sendMem = (struct ncclSendMem*) NCCL_NET_MAP_GET_POINTER(map, cpu, sendMem);
  resources->recvMem = (struct ncclRecvMem*) NCCL_NET_MAP_GET_POINTER(map, cpu, recvMem);
  for (int i = 0; i < NCCL_STEPS; i++) resources->recvMem->connFifo[i].size = -1;
  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
    resources->buffers[p] = NCCL_NET_MAP_GET_POINTER(map, cpu, buffs[p]);
    if (resources->buffers[p]) {
#if CUDA_VERSION >= 11070
      /* DMA-BUF support */
      int type = NCCL_NET_MAP_DEV_MEM(map, buffs[p]) ? NCCL_PTR_CUDA : NCCL_PTR_HOST;
      if (type == NCCL_PTR_CUDA && resources->useDmaBuf) {
        int dmabuf_fd;
        CUCHECK(cuMemGetHandleForAddressRange((void *)&dmabuf_fd, (CUdeviceptr)resources->buffers[p], resources->buffSizes[p], CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, getHandleForAddressRangeFlags(resources->useGdr)));
        NCCLCHECK(proxyState->ncclNet->regMrDmaBuf(resources->netRecvComm, resources->buffers[p], resources->buffSizes[p], type, 0ULL, dmabuf_fd, &resources->mhandles[p]));
        (void)close(dmabuf_fd);
      } else // FALL-THROUGH to nv_peermem GDR path
#endif
      {
        NCCLCHECK(proxyState->ncclNet->regMr(resources->netRecvComm, resources->buffers[p], resources->buffSizes[p], NCCL_NET_MAP_DEV_MEM(map, buffs[p]) ? NCCL_PTR_CUDA : NCCL_PTR_HOST, &resources->mhandles[p]));
      }

      // Copy the mhandle dptr
      if (resources->netDeviceType != NCCL_NET_DEVICE_HOST && proxyState->ncclNet->getDeviceMr)
        NCCLCHECK(proxyState->ncclNet->getDeviceMr(resources->netRecvComm, resources->mhandles[p], &connection->mhandles[p]));
    }
  }

  //NCCLCHECK(netDumpMap(map));
  if (respSize != sizeof(struct connectMap)) return ncclInternalError;
  memcpy(respBuff, map, sizeof(struct connectMap));
  return ncclSuccess;
}

static ncclResult_t sendProxyFree(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState) {
  struct sendNetResources* resources = (struct sendNetResources*)(connection->transportResources);
  if (connection->state == connSharedInitialized) { // NVB Preconnect
    NCCLCHECK(sharedNetBuffersDestroy(proxyState, connection->tpLocalRank, 0, connection));
    return ncclSuccess;
  }

  if (connection->state == connConnected) {
    while (!ncclIntruQueueEmpty(&connection->proxyMemHandleQueue)) {
      struct proxyMemHandle* memHandle = ncclIntruQueueDequeue(&connection->proxyMemHandleQueue);
      NCCLCHECK(proxyState->ncclNet->deregMr(resources->netSendComm, memHandle->handle));
      free(memHandle);
    }

    for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
      if (resources->buffers[p]) {
        NCCLCHECK(proxyState->ncclNet->deregMr(resources->netSendComm, resources->mhandles[p]));
      }
    }
    struct connectMapMem* mems = resources->map.mems;
    if (resources->map.sameProcess) {
      NCCLCHECK(ncclCudaHostFree(mems[NCCL_NET_MAP_HOSTMEM].cpuPtr));
    } else {
      NCCLCHECK(ncclShmIpcClose(&mems[NCCL_NET_MAP_HOSTMEM].createDesc));
    }
    NCCLCHECK(ncclCudaFree(mems[NCCL_NET_MAP_DEVMEM].cpuPtr, proxyState->memManager));
    if (!resources->map.sameProcess || ncclCuMemEnable()) {
      // cuMem API support
      if (mems[NCCL_NET_MAP_DEVMEM].size) {
        NCCLCHECK(ncclP2pFreeShareableBuffer(&mems[NCCL_NET_MAP_DEVMEM].ipcDesc));
      }
    }
    if (mems[NCCL_NET_MAP_GDCMEM].cpuPtr) NCCLCHECK(ncclGdrCudaFree(resources->gdrDesc, proxyState->memManager));
    if (resources->shared) {
      NCCLCHECK(sharedNetBuffersDestroy(proxyState, resources->tpLocalRank, 0, connection));
      if (resources->maxRecvs > 1 && ncclParamNetSharedComms()) {
        struct ncclSharedNetComms* comms = proxyState->progressState.netComms[resources->netDev]+resources->tpRemoteRank;
        comms->sendRefCount[resources->channelId]--;
        if (comms->sendRefCount[resources->channelId] == 0) NCCLCHECK(proxyState->ncclNet->closeSend(comms->sendComm[resources->channelId]));
      } else {
        NCCLCHECK(proxyState->ncclNet->closeSend(resources->netSendComm));
      }
    } else {
      NCCLCHECK(proxyState->ncclNet->closeSend(resources->netSendComm));
    }
  }

  if (resources) free(resources);
  return ncclSuccess;
}

static ncclResult_t recvProxyFree(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState) {
  struct recvNetResources* resources = (struct recvNetResources*)(connection->transportResources);
  if (connection->state == connSharedInitialized) { // NVB Preconnect
    NCCLCHECK(sharedNetBuffersDestroy(proxyState, connection->tpLocalRank, 1, connection));
    return ncclSuccess;
  }

  if (connection->state == connConnected) {
    while (!ncclIntruQueueEmpty(&connection->proxyMemHandleQueue)) {
      struct proxyMemHandle* memHandle = ncclIntruQueueDequeue(&connection->proxyMemHandleQueue);
      NCCLCHECK(proxyState->ncclNet->deregMr(resources->netRecvComm, memHandle->handle));
      free(memHandle);
    }

    for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
      if (resources->buffers[p]) {
        NCCLCHECK(proxyState->ncclNet->deregMr(resources->netRecvComm, resources->mhandles[p]));
      }
    }
    struct connectMapMem* mems = resources->map.mems;
    NCCLCHECK(ncclCudaHostFree(mems[NCCL_NET_MAP_HOSTMEM].cpuPtr));
    NCCLCHECK(ncclCudaFree(mems[NCCL_NET_MAP_DEVMEM].cpuPtr, proxyState->memManager));
    if (!resources->map.sameProcess || ncclCuMemEnable()) {
      // cuMem API support
      if (mems[NCCL_NET_MAP_DEVMEM].size) {
        NCCLCHECK(ncclP2pFreeShareableBuffer(&mems[NCCL_NET_MAP_DEVMEM].ipcDesc));
      }
    }
    if (mems[NCCL_NET_MAP_GDCMEM].cpuPtr) NCCLCHECK(ncclGdrCudaFree(resources->gdrDesc, proxyState->memManager));
    if (resources->shared) {
      NCCLCHECK(sharedNetBuffersDestroy(proxyState, resources->tpLocalRank, 1, connection));
      if (resources->maxRecvs > 1 && ncclParamNetSharedComms()) {
        struct ncclSharedNetComms* comms = proxyState->progressState.netComms[resources->netDev] + resources->tpRemoteProxyRank;
        comms->recvRefCount[resources->channelId]--;
        if (comms->recvRefCount[resources->channelId] == 0) NCCLCHECK(proxyState->ncclNet->closeRecv(comms->recvComm[resources->channelId]));
      } else {
        NCCLCHECK(proxyState->ncclNet->closeRecv(resources->netRecvComm));
      }
    } else {
      NCCLCHECK(proxyState->ncclNet->closeRecv(resources->netRecvComm));
    }
  }

  if (resources) free(resources);
  return ncclSuccess;
}

static_assert(NCCL_STEPS <= NCCL_NET_MAX_REQUESTS, "Not enough net requests to cover for steps");

static ncclResult_t sendProxyProgress(struct ncclProxyState* proxyState, struct ncclProxyArgs* args) {
  int checkedNetAttr = 0;
  if (args->state == ncclProxyOpReady) {
    for (int s=0; s<args->nsubs; s++) {
      struct ncclProxySubArgs* sub = args->subs+s;
      struct sendNetResources* resources = (struct sendNetResources*) (sub->connection->transportResources);
      // Round to next multiple of sliceSteps
      sub->base = ROUNDUP(resources->step, args->chunkSteps);
      // Set step base for next op
      resources->step = sub->base + sub->nsteps;
      sub->posted = sub->transmitted = sub->done = 0;
      sub->phase1WindowCfgLogged = 0;
      sub->phase1SendWstall = 0;
      sub->phase1RecvWstall = 0;
      sub->phase2WindowCfgLogged = 0;
      sub->phase2RecvWstall = 0;
      sub->phase3WindowCfgLogged = 0;
      sub->phase3RecvWstall = 0;
      sub->phase4WindowCfgLogged = 0;
      sub->phase4RecvWstall = 0;
      sub->phase6RateCfgLogged = 0;
      sub->phase6RateStall = 0;
      sub->phase6Tokens = 0.0;
      sub->phase6LastRefillNs = 0;
      sub->phase3CurrentW = 0;
      sub->phase3LastLoggedW = 0;
      sub->phase3HiCount = 0;
      sub->phase3LoCount = 0;
      sub->phase3WarmupCount = 0;
      sub->phase3CtrlStep = 0;
      sub->phase3DelayBaseNs = 0;
      sub->phase3DelayEwmaNs = 0;
      sub->phase3LastDelayNs = 0;
      memset(sub->phase3PostTs, 0, sizeof(sub->phase3PostTs));
      ncclProfilerRecordProxyOpEventState(s, args, ncclProfilerProxyOpInProgress_v4);
      if (!sub->reg)
        sub->sendMhandle = resources->mhandles[args->protocol];
    }
    args->state = ncclProxyOpProgress;
  }
  args->idle = 1;
  if (args->state == ncclProxyOpProgress) {
    int p = args->protocol;
    int wBase = phase1WindowBaseDepth(args);
    int wCfg = phase1WindowCfg();
    int wEff = phase1WindowEff(args);
    int sendDepth = wBase;
    for (int s=0; s<args->nsubs; s++) {
      struct ncclProxySubArgs* sub = args->subs+s;
      int postedStepId = sub->posted;
      int transmittedStepId = sub->transmitted;
      int doneStepId = sub->done;
      if (sub->done == sub->nsteps) continue;
      struct sendNetResources* resources = (struct sendNetResources*) (sub->connection->transportResources);
      volatile struct ncclConnFifo* connFifo = (volatile struct ncclConnFifo*)resources->recvMem->connFifo;
      int stepSize = resources->buffSizes[p] / NCCL_STEPS;
      char* localBuff = NCCL_NET_MAP_GET_POINTER(&resources->map, cpu, buffs[p]);
      // Post buffers to the GPU
      if (sub->posted < sub->nsteps && sub->posted < sub->done + sendDepth) {
        sub->phase1SendWstall = 0;
        ncclProfilerStartSendProxyStepEvent(s, args, postedStepId);
        int buffSlot = (sub->base+sub->posted)%NCCL_STEPS;
        if (resources->shared) {
          if (!sub->reg) {
            int sharedBuffSlot = sub->posted%sendDepth;
            int offset;
            NCCLCHECK(sharedBuffersGet(proxyState, sub->channelId, sharedBuffSlot*args->nsubs+s, &offset, NULL));
            resources->recvMem->connFifo[buffSlot].offset = offset;
            std::atomic_thread_fence(std::memory_order_seq_cst);
          }
          volatile uint64_t* sendHead = resources->gdcSync ? resources->gdcSync : &resources->sendMem->head;
          sub->posted += args->sliceSteps;
          *sendHead = sub->base + sub->posted - NCCL_STEPS;
          if (resources->gdcSync) wc_store_fence(); // Flush out WC write
        } else {
          sub->posted += args->sliceSteps;
        }
        ncclProfilerRecordProxyStepEventState(s, args, postedStepId, ncclProfilerProxyStepSendGPUWait);
        args->idle = 0;
        continue;
      }
      // Check whether we received data from the GPU and send it to the network
      if (sub->transmitted < sub->posted && sub->transmitted < sub->done + NCCL_STEPS) {
        int buffSlot = (sub->base+sub->transmitted)%NCCL_STEPS;
        volatile uint64_t* recvTail = &resources->recvMem->tail;
        uint64_t tail = sub->base + sub->transmitted;
        if (connFifo[buffSlot].size != -1 && (*recvTail > tail || p == NCCL_PROTO_LL)) {
          // We have something to receive, let's check if it's completely ready.
          int size = connFifo[buffSlot].size;
          bool shared = (p == NCCL_PROTO_SIMPLE) && resources->shared;
          char* buff = shared ? localBuff+connFifo[buffSlot].offset : localBuff+buffSlot*stepSize;
          int ready = 1;
          if (p == NCCL_PROTO_LL128) {
            ready = resources->useGdr;
            if (!ready) {
              // When data is in sysmem, we need to wait until all flags are correct since the GPU only
              // called threadfence()
              uint64_t flag = sub->base+sub->transmitted+1;
              int nFifoLines = DIVUP(connFifo[buffSlot].size, sizeof(uint64_t)*NCCL_LL128_LINEELEMS);
              volatile uint64_t* lines = (volatile uint64_t*)buff;
              ready = 1;
              for (int i=0; i<nFifoLines; i++) {
                if (lines[i*NCCL_LL128_LINEELEMS+NCCL_LL128_DATAELEMS] != flag) { ready = 0; break; }
              }
            }
          } else if (p == NCCL_PROTO_LL) {
            uint32_t flag = NCCL_LL_FLAG(sub->base+sub->transmitted+1);
            int nFifoLines = DIVUP(size, sizeof(union ncclLLFifoLine));
            union ncclLLFifoLine* lines = (union ncclLLFifoLine*)buff;
            for (int i=0; i<nFifoLines; i++) {
              volatile uint32_t *f1 = &lines[i].flag1;
              volatile uint32_t *f2 = &lines[i].flag2;
              if (f1[0] != flag || f2[0] != flag) { ready = 0; break; }
            }
          } else if (p == NCCL_PROTO_SIMPLE) {
            if (resources->shared) {
              buff = sub->reg ? (char*)sub->sendbuff + sub->transmitted * NCCL_MAX_NET_SIZE : localBuff + resources->recvMem->connFifo[buffSlot].offset;
            } else if (sub->reg) {
              size_t sendSize;
              sub->ringAlgo->getNextSendAddr(sub->transmitted, (uint8_t**)&buff, &sendSize, &sub->sendMhandle);
              assert(sendSize == size);
            }
          }
          if (ready) {
            ncclProfilerRecordProxyStepEventState(s, args, transmittedStepId, ncclProfilerProxyStepSendPeerWait_v4);
            // Data is ready, try to send.
            // Coverity complains about the size here as pointing to an out-of-scope temporary.  Which is nonsense,
            // since size is a plain integer.
            // coverity[use_invalid:FALSE]
            void* phandle = &sub->pHandles[DIVUP(transmittedStepId, args->sliceSteps)%NCCL_STEPS];
            if (!checkedNetAttr++)
              setXferNetAttrs(proxyState, args, 1);
            NCCLCHECK(proxyState->ncclNet->isend(resources->netSendComm, buff, size, resources->tpRank, sub->sendMhandle, phandle, sub->requests+buffSlot));
            if (sub->requests[buffSlot] != NULL) {
              TRACE(NCCL_NET, "sendProxy [%ld/%d/%d] Isend posted, req %p, buff %p, size %d, proto %d, myRank %d, channelId %d, mhandle %p", sub->transmitted, buffSlot, sub->nsteps, sub->requests[buffSlot], buff, size, p, proxyState->tpRank, sub->channelId, sub->sendMhandle);
              sub->transSize = size;
              phase0ProxyLog(proxyState, args, sub, "PROXY_SEND_POST", buffSlot, size, wBase, wCfg, wEff);
              sub->transmitted += args->sliceSteps;
              ncclProfilerRecordProxyStepEventState(s, args, transmittedStepId, ncclProfilerProxyStepSendWait);
              args->idle = 0;
              continue;
            }
          }
        }
      }
      // Check whether the network has completed some send operations.
      if (sub->done < sub->transmitted) {
        int done;
        int size;
        int buffSlot = (sub->base+sub->done)%NCCL_STEPS;
        NCCLCHECK(proxyState->ncclNet->test(sub->requests[buffSlot], &done, &size));
        if (done) {
          // Make sure size is reset to -1 before we update the head.
          connFifo[buffSlot].size = -1;
          std::atomic_thread_fence(std::memory_order_seq_cst);
          TRACE(NCCL_NET, "sendProxy [%ld/%d/%d] request %p done", sub->done, buffSlot, sub->nsteps, sub->requests[buffSlot]);
          phase0ProxyLog(proxyState, args, sub, "PROXY_SEND_DONE", buffSlot, size, wBase, wCfg, wEff);
          sub->done += args->sliceSteps;
          ncclProfilerStopProxyStepEvent(s, args, doneStepId);

          if (resources->shared == 0) {
            volatile uint64_t* sendHead = resources->gdcSync ? resources->gdcSync : &resources->sendMem->head;
            *sendHead = sub->base + sub->done;
            if (resources->gdcSync) wc_store_fence(); // Flush out WC write
          }
          args->idle = 0;
          if (sub->done == sub->nsteps) {
            args->done++;
            if (sub->ringAlgo && sub->ringAlgo->decRefCount() == 0) delete sub->ringAlgo;
            sub->ringAlgo = NULL;
          }
        }
      }
    }
    if (args->done == args->nsubs) {
      for (int s=0; s<args->nsubs; s++) {
        ncclProfilerStopProxyOpEvent(s, args);
      }
      args->state = ncclProxyOpNone;
    }
  }
  return ncclSuccess;
}

static ncclResult_t recvProxyProgress(struct ncclProxyState* proxyState, struct ncclProxyArgs* args) {
  int checkedNetAttr = 0;
  if (args->state == ncclProxyOpReady) {
    // Initialize subs and group them by same recvComm.
    args->phase4MaxDepthLogged = 0;
    void* recvComm;
    int groupSize = 0;
    int maxRecvs = 1;
    for (int s=0; s<args->nsubs; s++) {
      struct ncclProxySubArgs* sub = args->subs+s;
      if (groupSize == maxRecvs) {
        groupSize = 0;
      } else if (s>0) { // Find next sub with the same recvComm
        int next;
        for (next=s; next<args->nsubs; next++) {
          struct recvNetResources* nextRes = (struct recvNetResources*) (args->subs[next].connection->transportResources);
          if (nextRes->netRecvComm == recvComm) break;
        }
        if (next == args->nsubs) { // Not found
          groupSize = 0;
        } else if (s != next) { // We found a sub later with the same recvComm ; swap subs
          struct ncclProxySubArgs temp;
          memcpy(&temp, sub, sizeof(struct ncclProxySubArgs));
          memcpy(sub, args->subs+next, sizeof(struct ncclProxySubArgs));
          memcpy(args->subs+next, &temp, sizeof(struct ncclProxySubArgs));
        }
      }
      groupSize++;
      struct recvNetResources* resources = (struct recvNetResources*) (sub->connection->transportResources);
      maxRecvs = resources->maxRecvs;
      recvComm = resources->netRecvComm;
      // Round to next multiple of sliceSteps
      sub->base = ROUNDUP(resources->step, args->chunkSteps);
      // Set step base for next op
      resources->step = sub->base + sub->nsteps;
      sub->posted = sub->received = sub->transmitted = sub->done = 0;
      sub->regBufferReady = 0;
      sub->phase1WindowCfgLogged = 0;
      sub->phase1SendWstall = 0;
      sub->phase1RecvWstall = 0;
      sub->phase2WindowCfgLogged = 0;
      sub->phase2RecvWstall = 0;
      sub->phase3WindowCfgLogged = 0;
      sub->phase3RecvWstall = 0;
      sub->phase4WindowCfgLogged = 0;
      sub->phase4RecvWstall = 0;
      sub->phase6RateCfgLogged = 0;
      sub->phase6RateStall = 0;
      sub->phase6Tokens = 0.0;
      sub->phase6LastRefillNs = 0;
      sub->phase7RateCfgLogged = 0;
      sub->phase7RateStall = 0;
      sub->phase7ControlActive = 0;
      sub->phase7Tokens = 0.0;
      sub->phase7LastRefillNs = 0;
      sub->phase7ObserveStartNs = 0;
      sub->phase7ObservedPosts = 0.0;
      sub->phase7BaselineRatePerMs = 0.0;
      sub->phase7TargetRatePerMs = 0.0;
      sub->phase7TargetBurst = 0.0;
      sub->phase9LastPostNs = 0;
      sub->phase9PostSeq = 0;
      sub->phase3CurrentW = 0;
      sub->phase3LastLoggedW = 0;
      sub->phase3HiCount = 0;
      sub->phase3LoCount = 0;
      sub->phase3WarmupCount = 0;
      sub->phase3CtrlStep = 0;
      sub->phase3DelayBaseNs = 0;
      sub->phase3DelayEwmaNs = 0;
      sub->phase3LastDelayNs = 0;
      memset(sub->phase3PostTs, 0, sizeof(sub->phase3PostTs));
      memset(sub->phase5PostTs, 0, sizeof(sub->phase5PostTs));
      memset(sub->phase5PostProgressCall, 0, sizeof(sub->phase5PostProgressCall));
      for (int i=0; i<groupSize; i++) sub[-i].groupSize = groupSize;
      appendix2RecvGroupLog(proxyState, args, sub, s - groupSize + 1, groupSize, maxRecvs);
      ncclProfilerRecordProxyOpEventState(s, args, ncclProfilerProxyOpInProgress_v4);
      if (!sub->reg)
        sub->recvMhandle = resources->mhandles[args->protocol];
    }
    args->phase5RecvProxyCalls = 0;
    args->phase5LastRecvProxyNs = 0;
    args->state = ncclProxyOpProgress;
  }
  args->idle = 1;
  if (args->state == ncclProxyOpProgress) {
    int p = args->protocol;
    int wBase = phase1WindowBaseDepth(args);
    int wCfg = phase1WindowCfg();
    int phase4Enabled = ncclParamPhase4Enable();
    double phase4WRaw = phase4WindowRaw();
    int phase6RateEnabled = phase6Enabled();
    double phase6RateRaw = phase6PostRateRaw();
    double phase6BurstRaw = phase6PostBurstRaw();
    int phase7RateEnabled = phase7Enabled();
    double phase7RatioPct = phase7RateRatioRaw();
    double phase7ObserveMs = phase7ObserveMsRaw();
    double phase7BurstWindowMs = phase7BurstWindowMsRaw();
    double phase7BurstFloorPosts = phase7BurstFloorPostsRaw();
    if (ncclParamPhase5Log() != 0) {
      uint64_t nowNs = clockNano();
      uint64_t deltaNs = args->phase5LastRecvProxyNs ? nowNs - args->phase5LastRecvProxyNs : 0;
      args->phase5RecvProxyCalls++;
      args->phase5LastRecvProxyNs = nowNs;
      phase5RecvProgressLog(proxyState, args, wBase, phase4Enabled, phase4WRaw, deltaNs);
    }
    for (int s=0; s<args->nsubs; s+=args->subs[s].groupSize) {
      struct ncclProxySubArgs* subGroup = args->subs+s;
      int subCount = 0;
      void* ptrs[NCCL_PROXY_MAX_SUBS];
      size_t sizes[NCCL_PROXY_MAX_SUBS];
      int tags[NCCL_PROXY_MAX_SUBS];
      void* mhandles[NCCL_PROXY_MAX_SUBS];
      void* phandles[NCCL_PROXY_MAX_SUBS];
      for (int i=0; i<subGroup->groupSize; i++) {
        struct ncclProxySubArgs* sub = subGroup + i;
        int postedStepId = sub->posted;
        if (sub->posted < sub->nsteps) {
          struct recvNetResources* resources = (struct recvNetResources*) (sub->connection->transportResources);
          struct phase2WindowDecision semanticDecision = phase2SelectWindow(proxyState, args, sub);
          struct phase3WindowDecision phase3Decision = phase3SnapshotWindow(sub, &semanticDecision);
          int wEff = wBase;
          int slotDepth = wBase;
          phase4ProxyMaxDepthLog(proxyState, args, sub, wBase, phase4Enabled, phase4WRaw);
          if (phase4Enabled && phase4WRaw > 0.0) {
            wEff = phase4WindowEff(proxyState, args, sub, phase4WRaw);
            phase4ProxyWindowCfgLog(proxyState, args, sub, wBase, phase4WRaw, wEff);
          } else if (wCfg > 0) {
            wEff = phase1WindowEff(args);
          } else if (phase3Decision.enabled) {
            wEff = phase3Decision.wEff;
            phase3ProxyWindowCfgLog(proxyState, args, sub, &phase3Decision);
          } else if (semanticDecision.enabled) {
            wEff = semanticDecision.wEff;
            phase2ProxyWindowCfgLog(proxyState, args, sub, &semanticDecision);
          } else {
            phase1ProxyWindowCfgLog(proxyState, args, sub, resources->shared, wBase, wCfg, wEff);
          }
          if (sub->posted >= sub->done + wBase) {
            if (phase3Decision.enabled && wCfg == 0) {
              phase3ProxyRecvWstallLog(proxyState, args, sub, (sub->base+sub->posted)%NCCL_STEPS, &phase3Decision);
            } else if (semanticDecision.enabled && wCfg == 0) {
              phase2ProxyRecvWstallLog(proxyState, args, sub, (sub->base+sub->posted)%NCCL_STEPS, &semanticDecision);
            } else {
              phase1ProxyWstallLog(proxyState, args, sub, "PROXY_RECV_WSTALL", (sub->base+sub->posted)%NCCL_STEPS, wBase, wCfg, wBase, &sub->phase1RecvWstall);
            }
            subCount = 0;
            break;
          }
          if (phase4Enabled && phase4WRaw > 0.0 && sub->posted >= sub->received + wEff) {
            phase4ProxyWstallLog(proxyState, args, sub, "PROXY_RECV_WSTALL", (sub->base+sub->posted)%NCCL_STEPS, phase4WRaw, wEff, &sub->phase4RecvWstall);
            phase5RecvEventLog(proxyState, args, sub, "PROXY_RECV_WSTALL", (sub->base + sub->posted) % NCCL_STEPS, 0, wBase, phase4WRaw, wEff, 0, 0, 0);
            subCount = 0;
            break;
          }
          sub->phase1RecvWstall = 0;
          sub->phase2RecvWstall = 0;
          sub->phase3RecvWstall = 0;
          sub->phase4RecvWstall = 0;
          ncclProfilerStartRecvProxyStepEvent(s+i, args, postedStepId);
          int stepSize = resources->buffSizes[p] / NCCL_STEPS;
          char* localBuff = NCCL_NET_MAP_GET_POINTER(&resources->map, cpu, buffs[p]);
          int buffSlot = (sub->base+sub->posted)%NCCL_STEPS;
          volatile struct ncclConnFifo* connFifo = (volatile struct ncclConnFifo*)resources->recvMem->connFifo;
          if (p == NCCL_PROTO_SIMPLE) {
            if (resources->shared) {
              if (sub->reg) {
                // Wait until CUDA kernel has started before we access the user buffer directly.
                if (!sub->regBufferReady && connFifo[sub->base % NCCL_STEPS].size == -1) continue;
                sub->regBufferReady = 1;
                ptrs[subCount] = sub->recvbuff + sub->posted * NCCL_MAX_NET_SIZE;
                sizes[subCount] = std::min(NCCL_MAX_NET_SIZE, (ssize_t)(sub->nbytes - sub->posted * NCCL_MAX_NET_SIZE));
              } else {
                // Keep shared buffer slot indexing stable across sender/receiver.
                // Dynamic W only gates how far the receiver can advance.
                int sharedBuffSlot = sub->posted % slotDepth;
                int offset;
                NCCLCHECK(sharedBuffersGet(proxyState, sub->channelId, sharedBuffSlot * args->nsubs + s + i, &offset, sizes + subCount));
                connFifo[buffSlot].offset = offset;
                ptrs[subCount] = localBuff + offset;
              }
            } else {
              if (sub->reg) {
                if (!sub->regBufferReady && connFifo[sub->base % NCCL_STEPS].size == -1) continue;
                sub->regBufferReady = 1;
                sub->ringAlgo->getNextRecvAddr(sub->posted, (uint8_t**)&ptrs[subCount], &sizes[subCount], &sub->recvMhandle);
              } else {
                ptrs[subCount] = localBuff + buffSlot * stepSize;
                sizes[subCount] = stepSize * args->sliceSteps;
              }
            }
          } else {
            ptrs[subCount] = localBuff+buffSlot*stepSize;
            sizes[subCount] = stepSize*args->sliceSteps;
          }
          // if (sub->nbytes < sizes[subCount]) sizes[subCount] = sub->nbytes;
          tags[subCount] = resources->tpRemoteRank;
          mhandles[subCount] = sub->recvMhandle;
          phandles[subCount] = &sub->pHandles[DIVUP(postedStepId, args->sliceSteps)%NCCL_STEPS];
          subCount++;
        }
      }
      if (subCount) {
        uint64_t step = subGroup->posted;
        struct recvNetResources* resources = (struct recvNetResources*) (subGroup->connection->transportResources);
        void** requestPtr = subGroup->requests+(step%NCCL_STEPS);
        if (phase7RateEnabled && phase7RatioPct > 0.0) {
          struct ncclProxySubArgs* leader = subGroup;
          uint64_t nowNs = clockNano();
          if (!leader->phase7RateCfgLogged) {
            leader->phase7ObserveStartNs = nowNs;
            phase7RateCfgLog(proxyState, args, leader, phase7RatioPct, phase7ObserveMs, phase7BurstWindowMs, phase7BurstFloorPosts);
            leader->phase7RateCfgLogged = 1;
          }
          int postCost = subCount;
          if (!leader->phase7ControlActive) {
            if (leader->phase7ObserveStartNs == 0) leader->phase7ObserveStartNs = nowNs;
            uint64_t observedElapsedNs = nowNs - leader->phase7ObserveStartNs;
            if (observedElapsedNs >= (uint64_t)(phase7ObserveMs * 1000000.0) && leader->phase7ObservedPosts > 0.0) {
              double observedElapsedMs = (double)observedElapsedNs / 1000000.0;
              leader->phase7BaselineRatePerMs = observedElapsedMs > 0.0 ? (leader->phase7ObservedPosts / observedElapsedMs) : 0.0;
              leader->phase7TargetRatePerMs = leader->phase7BaselineRatePerMs * (phase7RatioPct * 0.01);
              leader->phase7TargetBurst = std::max(std::max((double)leader->groupSize, phase7BurstFloorPosts), leader->phase7TargetRatePerMs * phase7BurstWindowMs);
              leader->phase7Tokens = leader->phase7TargetBurst;
              leader->phase7LastRefillNs = nowNs;
              leader->phase7ControlActive = 1;
              phase7BaselineLog(proxyState, args, leader, observedElapsedNs, leader->phase7ObservedPosts, leader->phase7BaselineRatePerMs, leader->phase7TargetRatePerMs, leader->phase7TargetBurst);
            }
          }
          if (!leader->phase7ControlActive) {
            leader->phase7ObservedPosts += (double)postCost;
          } else {
            uint64_t elapsedNs = leader->phase7LastRefillNs ? (nowNs - leader->phase7LastRefillNs) : 0;
            double tokensBefore = leader->phase7Tokens;
            if (elapsedNs > 0) {
              leader->phase7Tokens = std::min(leader->phase7TargetBurst, leader->phase7Tokens + leader->phase7TargetRatePerMs * ((double)elapsedNs / 1000000.0));
            }
            leader->phase7LastRefillNs = nowNs;
            if (leader->phase7Tokens + 1.0e-12 < (double)postCost) {
              phase7RateDecisionLog(proxyState, args, leader, "RATE_STALL", elapsedNs, postCost, phase7RatioPct, leader->phase7BaselineRatePerMs, leader->phase7TargetRatePerMs, leader->phase7TargetBurst, tokensBefore, leader->phase7Tokens);
              leader->phase7RateStall = 1;
              continue;
            }
            leader->phase7RateStall = 0;
            leader->phase7Tokens -= (double)postCost;
            phase7RateDecisionLog(proxyState, args, leader, "RATE_ALLOW", elapsedNs, postCost, phase7RatioPct, leader->phase7BaselineRatePerMs, leader->phase7TargetRatePerMs, leader->phase7TargetBurst, tokensBefore, leader->phase7Tokens);
          }
        } else if (phase6RateEnabled && phase6RateRaw > 0.0 && phase6BurstRaw > 0.0) {
          struct ncclProxySubArgs* leader = subGroup;
          uint64_t nowNs = clockNano();
          if (!leader->phase6RateCfgLogged) {
            leader->phase6Tokens = phase6BurstRaw;
            leader->phase6LastRefillNs = nowNs;
            phase6RateCfgLog(proxyState, args, leader, phase6RateRaw, phase6BurstRaw);
            leader->phase6RateCfgLogged = 1;
          }
          uint64_t elapsedNs = leader->phase6LastRefillNs ? (nowNs - leader->phase6LastRefillNs) : 0;
          double tokensBefore = leader->phase6Tokens;
          if (elapsedNs > 0) {
            leader->phase6Tokens = std::min(phase6BurstRaw, leader->phase6Tokens + phase6RateRaw * ((double)elapsedNs / 1000000.0));
          }
          leader->phase6LastRefillNs = nowNs;
          int postCost = subCount;
          if (leader->phase6Tokens + 1.0e-12 < (double)postCost) {
            phase6RateDecisionLog(proxyState, args, leader, "RATE_STALL", elapsedNs, postCost, phase6RateRaw, phase6BurstRaw, tokensBefore, leader->phase6Tokens);
            leader->phase6RateStall = 1;
            continue;
          }
          leader->phase6RateStall = 0;
          leader->phase6Tokens -= (double)postCost;
          phase6RateDecisionLog(proxyState, args, leader, "RATE_ALLOW", elapsedNs, postCost, phase6RateRaw, phase6BurstRaw, tokensBefore, leader->phase6Tokens);
        }
        bool ignoreCompletion = ncclParamNetOptionalRecvCompletion() && ((args->protocol == NCCL_PROTO_LL128) || (args->protocol == NCCL_PROTO_LL)) && (subCount == 1);
        if (!checkedNetAttr++)
          setXferNetAttrs(proxyState, args, 0);
        if (ignoreCompletion) *requestPtr = (void *)NCCL_NET_OPTIONAL_RECV_COMPLETION;
        NCCLCHECK(proxyState->ncclNet->irecv(resources->netRecvComm, subCount, ptrs, sizes, tags, mhandles, phandles, requestPtr));
        if (*requestPtr) {
          struct ncclProxySubArgs* leader = subGroup;
          uint64_t phase9NowNs = clockNano();
          uint64_t phase9DeltaNs = leader->phase9LastPostNs ? (phase9NowNs - leader->phase9LastPostNs) : 0;
          double phase9InstRatePerMs = (phase9DeltaNs > 0) ? ((double)subCount / ((double)phase9DeltaNs / 1000000.0)) : 0.0;
          leader->phase9PostSeq += 1;
          phase9PostRateLog(proxyState, args, leader, phase9NowNs, subCount, phase9DeltaNs, phase9InstRatePerMs);
          leader->phase9LastPostNs = phase9NowNs;
          subGroup->recvRequestsCache[step%NCCL_STEPS] = *requestPtr;
          subGroup->recvRequestsSubCount = subCount;
          for (int i=0; i<subGroup->groupSize; i++) {
            struct ncclProxySubArgs* sub = subGroup+i;
            int postedStepId = sub->posted;
            struct phase2WindowDecision semanticDecision = phase2SelectWindow(proxyState, args, sub);
            struct phase3WindowDecision phase3Decision = phase3SnapshotWindow(sub, &semanticDecision);
            int wEff = (wCfg > 0) ? phase1WindowEff(args) : (phase3Decision.enabled ? phase3Decision.wEff : (semanticDecision.enabled ? semanticDecision.wEff : wBase));
            int buffSlot = (sub->base + sub->posted) % NCCL_STEPS;
            uint64_t postTsNs = clockNano();
            TRACE(NCCL_NET, "recvProxy [%ld/%d/%d] Irecv posted, buff %p, size %ld, myRank %d, channelId %d, mhandle %p", sub->posted, buffSlot, sub->nsteps, ptrs[i], sizes[i], proxyState->tpRank, sub->channelId, mhandles[i]);
            phase0ProxyLog(proxyState, args, sub, "PROXY_RECV_POST", buffSlot, sizes[i], wBase, wCfg, wEff);
            sub->phase3PostTs[buffSlot] = postTsNs;
            sub->phase5PostTs[buffSlot] = postTsNs;
            sub->phase5PostProgressCall[buffSlot] = args->phase5RecvProxyCalls;
            phase5RecvEventLog(proxyState, args, sub, "PROXY_RECV_POST", buffSlot, sizes[i], wBase, phase4WRaw, wEff, postTsNs, 0, 0);
            sub->posted += args->sliceSteps;
            ncclProfilerRecordProxyStepEventState(s+i, args, postedStepId, ncclProfilerProxyStepRecvWait);
          }
          args->idle = 0;
        }
      }
    }
    if (args->idle == 0) return ncclSuccess;

    for (int s=0; s<args->nsubs; s+=args->subs[s].groupSize) {
      struct ncclProxySubArgs* subGroup = args->subs+s;
      if (subGroup->posted > subGroup->received) {
        uint64_t step = subGroup->received;
        int done;
        void* ptrs[NCCL_PROXY_MAX_SUBS];
        int sizes[NCCL_PROXY_MAX_SUBS];
        void* mhandles[NCCL_PROXY_MAX_SUBS];
        for (int i=0; i<NCCL_PROXY_MAX_SUBS; i++) sizes[i] = 0;
        NCCLCHECK(proxyState->ncclNet->test(subGroup->requests[step%NCCL_STEPS], &done, sizes));
        if (done) {
          int needFlush = 0;
          int totalSize = 0;
          for (int i=0; i<NCCL_PROXY_MAX_SUBS; i++) totalSize += sizes[i];
          for (int i=0; i<subGroup->groupSize; i++) {
            struct ncclProxySubArgs* sub = subGroup + i;
            int receivedStepId = sub->received;
            int buffSlot = (sub->base + sub->received) % NCCL_STEPS;
            struct phase2WindowDecision semanticDecision = phase2SelectWindow(proxyState, args, sub);
            struct phase3WindowDecision phase3Decision = phase3SnapshotWindow(sub, &semanticDecision);
            int wEff = (wCfg > 0) ? phase1WindowEff(args) : (phase3Decision.enabled ? phase3Decision.wEff : (semanticDecision.enabled ? semanticDecision.wEff : wBase));
            struct recvNetResources* resources = (struct recvNetResources*)(sub->connection->transportResources);
            volatile struct ncclConnFifo* connFifo = (volatile struct ncclConnFifo*)resources->recvMem->connFifo;
            connFifo[buffSlot].size = -1;
            sub->transSize = sizes[i];
            uint64_t nowNs = clockNano();
            uint64_t delayNs = 0;
            if (sub->phase3PostTs[buffSlot] != 0 && nowNs >= sub->phase3PostTs[buffSlot]) {
              delayNs = nowNs - sub->phase3PostTs[buffSlot];
              sub->phase3LastDelayNs = delayNs;
              sub->phase3DelayEwmaNs = phase3UpdateDelayEwma(sub->phase3DelayEwmaNs, delayNs);
              if (sub->phase3DelayBaseNs == 0) sub->phase3DelayBaseNs = sub->phase3DelayEwmaNs;
            }
            phase0ProxyLog(proxyState, args, sub, "PROXY_RECV_NET_DONE", buffSlot, sizes[i], wBase, wCfg, wEff);
            {
              uint64_t phase5PostTsNs = sub->phase5PostTs[buffSlot];
              uint64_t phase5DelayNs = (phase5PostTsNs != 0 && nowNs >= phase5PostTsNs) ? (nowNs - phase5PostTsNs) : 0;
              uint64_t progressCallsSincePost = sub->phase5PostProgressCall[buffSlot] ? (args->phase5RecvProxyCalls - sub->phase5PostProgressCall[buffSlot]) : 0;
              phase5RecvEventLog(proxyState, args, sub, "PROXY_RECV_NET_DONE", buffSlot, sizes[i], wBase, phase4WRaw, wEff, phase5PostTsNs, phase5DelayNs, progressCallsSincePost);
            }
            sub->received += args->sliceSteps;
            ncclProfilerRecordProxyStepEventState(s+i, args, receivedStepId, ncclProfilerProxyStepRecvFlushWait);
            if (phase3Decision.enabled && wCfg == 0) {
              struct phase3WindowDecision updated = phase3UpdateController(sub, &semanticDecision);
              phase3ProxyPressureLog(proxyState, args, sub, &updated);
              phase3ProxyDecisionLog(proxyState, args, sub, &updated);
              phase3ProxyWindowCfgLog(proxyState, args, sub, &updated);
            }
            if (step < sub->nsteps) {
              struct recvNetResources* resources = (struct recvNetResources*) (sub->connection->transportResources);
              if (resources->useGdr) needFlush |= resources->needFlush;
            }
          }
          subGroup->requests[step%NCCL_STEPS] = NULL;
          if (totalSize > 0 && p == NCCL_PROTO_SIMPLE && needFlush) {
            // GDRCOPY support
            struct recvNetResources* resources = (struct recvNetResources*) (subGroup->connection->transportResources);
            if (resources->gdcFlush) {
#if defined (__x86_64__)
              // Force a PCI-E read from GPU memory
              asm volatile ("mov (%0), %%eax" :: "l"(resources->gdcFlush) : "%eax", "memory");
#else
              WARN("NET: GDR Flush only supported on x86_64");
              return ncclInternalError;
#endif
            } else {
              int subCount = 0;
              for (int i=0; i<subGroup->groupSize; i++) {
                struct ncclProxySubArgs* sub = subGroup + i;
                if (step < sub->nsteps) {
                  struct recvNetResources* resources = (struct recvNetResources*) (sub->connection->transportResources);
                  int stepSize = resources->buffSizes[p] / NCCL_STEPS;
                  char* localBuff = NCCL_NET_MAP_GET_POINTER(&resources->map, cpu, buffs[p]);
                  int buffSlot = (sub->base+sub->received-args->sliceSteps)%NCCL_STEPS;
                  if (resources->shared) {
                    ptrs[subCount] = sub->reg ? (char*)sub->recvbuff + step * NCCL_MAX_NET_SIZE : localBuff + resources->recvMem->connFifo[buffSlot].offset;
                  } else {
                    if (sub->reg) {
                      sub->ringAlgo->getNextRecvAddr(step, (uint8_t**)&ptrs[subCount], NULL, &sub->recvMhandle);
                    } else {
                      ptrs[subCount] = localBuff + buffSlot * stepSize;
                    }
                  }
                  mhandles[subCount] = sub->recvMhandle;
                  subCount++;
                }
              }
              struct recvNetResources* resources = (struct recvNetResources*) (subGroup->connection->transportResources);
              NCCLCHECK(proxyState->ncclNet->iflush(resources->netRecvComm, subCount, ptrs, sizes, mhandles, subGroup->requests+(step%NCCL_STEPS)));
            }
          }
          args->idle = 0;
        }
      }
    }
    if (args->idle == 0) return ncclSuccess;

    for (int s=0; s<args->nsubs; s+=args->subs[s].groupSize) {
      struct ncclProxySubArgs* subGroup = args->subs+s;
      if (subGroup->received > subGroup->transmitted) {
        uint64_t step = subGroup->transmitted;
        int done = 1;
        void* request = subGroup->requests[step%NCCL_STEPS];
        if (request) NCCLCHECK(proxyState->ncclNet->test(request, &done, NULL));
        if (done) {
          for (int i=0; i<subGroup->groupSize; i++) {
            struct ncclProxySubArgs* sub = subGroup + i;
            int transmittedStepId = sub->transmitted;
            struct phase2WindowDecision semanticDecision = phase2SelectWindow(proxyState, args, sub);
            struct phase3WindowDecision phase3Decision = phase3SnapshotWindow(sub, &semanticDecision);
            int wEff = (wCfg > 0) ? phase1WindowEff(args) : (phase3Decision.enabled ? phase3Decision.wEff : (semanticDecision.enabled ? semanticDecision.wEff : wBase));

            sub->transmitted += args->sliceSteps;
            ncclProfilerRecordProxyStepEventState(s+i, args, transmittedStepId, ncclProfilerProxyStepRecvGPUWait);
            if (step < sub->nsteps) {
              std::atomic_thread_fence(std::memory_order_seq_cst);
              struct recvNetResources* resources = (struct recvNetResources*) (sub->connection->transportResources);
              volatile uint64_t* recvTail = resources->gdcSync ? resources->gdcSync : &resources->recvMem->tail;
              *recvTail = sub->base + sub->transmitted;
              if (resources->gdcSync) wc_store_fence(); // Flush out WC write
            }
            phase0ProxyLog(proxyState, args, sub, "PROXY_RECV_VISIBLE", (sub->base + transmittedStepId) % NCCL_STEPS, sub->transSize, wBase, wCfg, wEff);
          }
          args->idle = 0;
        }
      }
    }
    if (args->idle == 0) return ncclSuccess;

    for (int s=0; s<args->nsubs; s+=args->subs[s].groupSize) {
      struct ncclProxySubArgs* subGroup = args->subs+s;
      for (int i=0; i<subGroup->groupSize; i++) {
        struct ncclProxySubArgs* sub = subGroup + i;
        if (sub->done == sub->nsteps) continue;
        if (sub->transmitted > sub->done) {
          struct recvNetResources* resources = (struct recvNetResources*) (sub->connection->transportResources);
          volatile uint64_t* sendHead = &resources->sendMem->head;
          uint64_t done = *sendHead;
          while (done > sub->base + sub->done &&
              // LL and LL128 can acknowledge 0-bytes send before they even happen. Don't go past what we transmitted.
              sub->transmitted > sub->done) {
            if (subGroup->recvRequestsCache[sub->done%NCCL_STEPS]) {
              // the multirecv requests are only cached in the first sub.
              if (proxyState->ncclNet->irecvConsumed)
                NCCLCHECK(proxyState->ncclNet->irecvConsumed(resources->netRecvComm, subGroup->recvRequestsSubCount, subGroup->recvRequestsCache[sub->done%NCCL_STEPS]));
              subGroup->recvRequestsCache[sub->done%NCCL_STEPS] = NULL;
            }
            int doneStepId = sub->done;
            struct phase2WindowDecision semanticDecision = phase2SelectWindow(proxyState, args, sub);
            struct phase3WindowDecision phase3Decision = phase3SnapshotWindow(sub, &semanticDecision);
            int wEff = (wCfg > 0) ? phase1WindowEff(args) : (phase3Decision.enabled ? phase3Decision.wEff : (semanticDecision.enabled ? semanticDecision.wEff : wBase));
            phase0ProxyLog(proxyState, args, sub, "PROXY_RECV_CONSUMED", (sub->base + sub->done) % NCCL_STEPS, sub->transSize, wBase, wCfg, wEff);
            sub->done += args->sliceSteps;
            ncclProfilerStopProxyStepEvent(s+i, args, doneStepId);
            if (phase3Decision.enabled && wCfg == 0) {
              struct phase3WindowDecision updated = phase3UpdateController(sub, &semanticDecision);
              phase3ProxyPressureLog(proxyState, args, sub, &updated);
              phase3ProxyDecisionLog(proxyState, args, sub, &updated);
              phase3ProxyWindowCfgLog(proxyState, args, sub, &updated);
            }
            args->idle = 0;
            if (sub->done == sub->nsteps) {
              args->done++;
              if (sub->ringAlgo && sub->ringAlgo->decRefCount() == 0) delete sub->ringAlgo;
              sub->ringAlgo = NULL;
              break;
            }
          }
        }
      }
    }
    if (args->done == args->nsubs) {
      args->state = ncclProxyOpNone;
      for (int s=0; s<args->nsubs; s++) {
        ncclProfilerStopProxyOpEvent(s, args);
      }
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclNetDeregBuffer(struct ncclComm* comm, struct ncclProxyConnector* proxyConn, void* handle) {
  NCCLCHECK(ncclProxyCallBlocking(comm, proxyConn, ncclProxyMsgDeregister, &handle, sizeof(void*), NULL, 0));
  INFO(NCCL_REG, "rank %d - deregistered net buffer handle %p", comm->rank, handle);
  return ncclSuccess;
}

static ncclResult_t netRegisterBuffer(ncclComm* comm, const void* userbuff, size_t buffSize, struct ncclConnector** peerConns, int nPeers, struct ncclReg* regRecord, int* outRegBufFlag, void** outHandle, int numSegments) {
  ncclResult_t ret = ncclSuccess;
  int gdrFlag = 1;

  if (regRecord) {
    for (int p = 0; p < nPeers; ++p) {
      struct ncclConnector* peerConn = peerConns[p];
      struct ncclProxyConnector* peerProxyConn = NULL;
      struct ncclRegNetHandles* netHandle = NULL;
      bool found = false;
      if (peerConn == NULL) continue;
      peerProxyConn = &peerConn->proxyConn;
      netHandle = regRecord->netHandleHead;
      while (netHandle) {
        if (netHandle->proxyConn == peerProxyConn) {
          found = true;
          break;
        }
        netHandle = netHandle->next;
      }
      if (found) {
        *outRegBufFlag = 1;
        outHandle[p] = netHandle->handle;
        INFO(NCCL_REG, "rank %d - NET reuse buffer %p size %ld (baseAddr %p size %ld) handle %p", comm->rank, userbuff, buffSize, (void*)regRecord->begAddr, regRecord->endAddr - regRecord->begAddr, netHandle->handle);
      } else {
        struct netRegInfo info = { regRecord->begAddr, regRecord->endAddr - regRecord->begAddr, numSegments};
        void* handle = NULL;

        if (peerConn->conn.flags & NCCL_DIRECT_NIC) {
          NCCLCHECKGOTO(ncclProxyCallBlocking(comm, peerProxyConn, ncclProxyMsgRegister, &info, sizeof(struct netRegInfo), &handle, sizeof(void*)), ret, fail);
          if (handle) {
            struct ncclRegNetHandles* netHandle;
            regRecord->state |= NET_REG_COMPLETE;
            NCCLCHECK(ncclCalloc(&netHandle, 1));
            netHandle->handle = handle;
            netHandle->proxyConn = peerProxyConn;
            netHandle->next = regRecord->netHandleHead;
            regRecord->netHandleHead = netHandle;
            outHandle[p] = handle;
            *outRegBufFlag = 1;
            INFO(NCCL_REG, "rank %d - NET register userbuff %p (handle %p), buffSize %ld", comm->rank, userbuff, handle, buffSize);
          } else {
            goto fail;
          }
        } else {
          gdrFlag = 0;
          goto fail;
        }
      }
    }
  }

exit:
  return ret;
fail:
  *outRegBufFlag = 0;
  INFO(NCCL_REG, "rank %d failed to NET register userbuff %p buffSize %ld GDR flag %d", comm->rank, userbuff, buffSize, gdrFlag);
  goto exit;
}

ncclResult_t ncclNetLocalRegisterBuffer(ncclComm* comm, const void* userbuff, size_t buffSize, struct ncclConnector** peerConns, int nPeers, int* outRegBufFlag, void** outHandle) {
  ncclResult_t ret = ncclSuccess;
  struct ncclReg *regRecord = NULL;
  bool isValid = false;
  void *base = NULL;
  size_t baseSize = 0;

  *outRegBufFlag = 0;
  if (comm && userbuff && buffSize > 0 && nPeers > 0) {
    NCCLCHECKGOTO(ncclRegFind(comm, userbuff, buffSize, &regRecord), ret, fail);
    NCCLCHECKGOTO(ncclRegLocalIsValid(regRecord, &isValid), ret, fail);
    if (isValid) {
      int numSegments = 0;
      NCCLCHECK(ncclCuMemGetAddressRange((CUdeviceptr) userbuff, buffSize, (CUdeviceptr*)&base, &baseSize, &numSegments));
      if (numSegments > 1 && !ncclParamMultiSegmentRegister()) goto exit;
      NCCLCHECKGOTO(netRegisterBuffer(comm, userbuff, buffSize, peerConns, nPeers, regRecord, outRegBufFlag, outHandle, numSegments), ret, fail);
    }
  }

exit:
  return ret;
fail:
  *outRegBufFlag = 0;
  goto exit;
}

struct ncclNetCleanupCallback {
  struct ncclCommCallback base;
  struct ncclComm *comm;
  struct ncclReg *reg;
};

static ncclResult_t cleanupNet(struct ncclComm* comm, struct ncclCommCallback* cb) {
  struct ncclNetCleanupCallback* obj = (struct ncclNetCleanupCallback*)cb;
  NCCLCHECK(ncclCommGraphDeregister(obj->comm, obj->reg));
  free(obj);
  return ncclSuccess;
}

ncclResult_t ncclNetGraphRegisterBuffer(ncclComm* comm, const void* userbuff, size_t buffSize, struct ncclConnector** peerConns, int nPeers, int* outRegBufFlag, void** outHandle, struct ncclIntruQueue<struct ncclCommCallback, &ncclCommCallback::next>* cleanupQueue, int* nCleanupQueueElts) {
  ncclResult_t ret = ncclSuccess;
  struct ncclNetCleanupCallback *record = NULL;
  struct ncclReg *regRecord = NULL;
  void *base = NULL;
  size_t baseSize = 0;

  *outRegBufFlag = 0;
  if (comm && userbuff && buffSize > 0 && nPeers > 0) {
    int numSegments = 0;
    NCCLCHECK(ncclCuMemGetAddressRange((CUdeviceptr) userbuff, buffSize, (CUdeviceptr*)&base, &baseSize, &numSegments));
    if (numSegments > 1 && !ncclParamMultiSegmentRegister()) goto exit;
    NCCLCHECKGOTO(ncclCommGraphRegister(comm, base, baseSize, (void**)&regRecord), ret, fail);
    NCCLCHECKGOTO(netRegisterBuffer(comm, userbuff, buffSize, peerConns, nPeers, regRecord, outRegBufFlag, outHandle, numSegments), ret, fail);
    if (*outRegBufFlag) {
      NCCLCHECKGOTO(ncclCalloc(&record, 1), ret, fail);
      record->base.fn = cleanupNet;
      record->comm = comm;
      record->reg = regRecord;
      ncclIntruQueueEnqueue(cleanupQueue, (struct ncclCommCallback*)record);
      if (nCleanupQueueElts) *nCleanupQueueElts += 1;
    } else {
      NCCLCHECKGOTO(ncclCommGraphDeregister(comm, regRecord), ret, fail);
    }
  }
exit:
  return ret;
fail:
  *outRegBufFlag = 0;
  goto exit;
}

static ncclResult_t sendProxyRegBuffer(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, void* reqBuff, int reqSize, void* respBuff, int respSize, int* done) {
  void* handle = NULL;
  struct netRegInfo* info = (struct netRegInfo*)reqBuff;
  int numSegments = info->numSegments;
  struct sendNetResources* resources = (struct sendNetResources*)(connection->transportResources);
  // The value of ret is ignored
  ncclResult_t ret;
  bool needReg = true;

  assert(reqSize == sizeof(struct netRegInfo));
  assert(respSize == sizeof(void*));

#if CUDART_VERSION >= 11070
  /* DMA-BUF support */
  if (resources->useDmaBuf) {
    int dmabuf_fd;
    CUCHECKGOTO(cuMemGetHandleForAddressRange((void*)&dmabuf_fd, (CUdeviceptr)info->buffer, info->size, CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, getHandleForAddressRangeFlags(resources->useGdr)), ret, peermem);
    NCCLCHECKGOTO(proxyState->ncclNet->regMrDmaBuf(resources->netSendComm, (void*)info->buffer, info->size, NCCL_PTR_CUDA, 0ULL, dmabuf_fd, &handle), ret, peermem);
    (void)close(dmabuf_fd);
    needReg = false;
  }
peermem:
#endif
  if (needReg) {
    // Non-dmabuf regMr does not support multiple physical segments
    if (numSegments > 1) {
      INFO(NCCL_NET|NCCL_REG, "Buffer %p (size %zu, numSegments %d) not registered as DMABuf is not available. Non-DMABuf registration currently does not support multiple segments.", (void *)info->buffer, info->size, numSegments);
      goto fail;
    } else {
      NCCLCHECKGOTO(proxyState->ncclNet->regMr(resources->netSendComm, (void*)info->buffer, info->size, NCCL_PTR_CUDA, &handle), ret, fail);
    }
  }

exit:
  if (handle) {
    struct proxyMemHandle* memHandle;
    NCCLCHECK(ncclCalloc(&memHandle, 1));
    memHandle->handle = handle;
    ncclIntruQueueEnqueue(&connection->proxyMemHandleQueue, memHandle);
  }
  memcpy(respBuff, (void*)&handle, sizeof(void*));
  *done = 1;
  return ncclSuccess;
fail:
  handle = NULL;
  goto exit;
}

static ncclResult_t recvProxyRegBuffer(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, void* reqBuff, int reqSize, void* respBuff, int respSize, int* done) {
  void* handle = NULL;
  struct netRegInfo* info = (struct netRegInfo*)reqBuff;
  int numSegments = info->numSegments;
  struct recvNetResources* resources = (struct recvNetResources*)(connection->transportResources);
  // The value of ret is ignored
  ncclResult_t ret;
  bool needReg = true;

  assert(reqSize == sizeof(struct netRegInfo));
  assert(respSize == sizeof(void*));

#if CUDART_VERSION >= 11070
  /* DMA-BUF support */
  if (resources->useDmaBuf) {
    int dmabuf_fd;
    CUCHECKGOTO(cuMemGetHandleForAddressRange((void*)&dmabuf_fd, (CUdeviceptr)info->buffer, info->size, CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, getHandleForAddressRangeFlags(resources->useGdr)), ret, peermem);
    NCCLCHECKGOTO(proxyState->ncclNet->regMrDmaBuf(resources->netRecvComm, (void*)info->buffer, info->size, NCCL_PTR_CUDA, 0ULL, dmabuf_fd, &handle), ret, peermem);
    (void)close(dmabuf_fd);
    needReg = false;
  }
peermem:
#endif
  if (needReg) {
    // Non-dmabuf regMr does not support multiple physical segments
    if (numSegments > 1) {
      INFO(NCCL_NET|NCCL_REG, "Buffer %p (size %zu, numSegments %d) not registered as DMABuf is not available. Non-DMABuf registration currently does not support multiple segments.", (void *)info->buffer, info->size, numSegments);
      goto fail;
    } else {
      NCCLCHECKGOTO(proxyState->ncclNet->regMr(resources->netRecvComm, (void*)info->buffer, info->size, NCCL_PTR_CUDA, &handle), ret, fail);
    }
  }

exit:
  if (handle) {
    struct proxyMemHandle* memHandle;
    NCCLCHECK(ncclCalloc(&memHandle, 1));
    memHandle->handle = handle;
    ncclIntruQueueEnqueue(&connection->proxyMemHandleQueue, memHandle);
  }
  memcpy(respBuff, (void*)&handle, sizeof(void*));
  *done = 1;
  return ncclSuccess;
fail:
  handle = NULL;
  goto exit;
}

static bool netHandleCmp(struct proxyMemHandle* a, struct proxyMemHandle* b) {
  return a->handle == b->handle;
}

static ncclResult_t sendProxyDeregBuffer(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, void* reqBuff, int reqSize, int* done) {
  void* handle;
  struct sendNetResources* resources = (struct sendNetResources*)(connection->transportResources);

  assert(reqSize == sizeof(void*));
  memcpy(&handle, reqBuff, sizeof(void*));
  if (handle) {
    struct proxyMemHandle memHandle = {};
    struct proxyMemHandle* deletedHandle;
    memHandle.handle = handle;
    deletedHandle = ncclIntruQueueDelete(&connection->proxyMemHandleQueue, &memHandle, netHandleCmp);
    free(deletedHandle);
  }
  NCCLCHECK(proxyState->ncclNet->deregMr(resources->netSendComm, handle));
  *done = 1;
  return ncclSuccess;
}

static ncclResult_t recvProxyDeregBuffer(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, void* reqBuff, int reqSize, int* done) {
  void* handle;
  struct recvNetResources* resources = (struct recvNetResources*)(connection->transportResources);

  assert(reqSize == sizeof(void*));
  memcpy(&handle, reqBuff, sizeof(void*));
  if (handle) {
    struct proxyMemHandle memHandle = {};
    struct proxyMemHandle* deletedHandle;
    memHandle.handle = handle;
    deletedHandle = ncclIntruQueueDelete(&connection->proxyMemHandleQueue, &memHandle, netHandleCmp);
    free(deletedHandle);
  }
  NCCLCHECK(proxyState->ncclNet->deregMr(resources->netRecvComm, handle));
  *done = 1;
  return ncclSuccess;
}

struct ncclTransport netTransport = {
  "NET",
  canConnect,
  { sendSetup, sendConnect, sendFree, proxySharedInit, sendProxySetup, sendProxyConnect, sendProxyFree, sendProxyProgress, sendProxyRegBuffer, sendProxyDeregBuffer },
  { recvSetup, recvConnect, recvFree, proxySharedInit, recvProxySetup, recvProxyConnect, recvProxyFree, recvProxyProgress, recvProxyRegBuffer, recvProxyDeregBuffer }
};
