/* ============================================================================
 * TMC3 PredGeom Only
 *
 * Simplified TMC3 flow:
 *   input PLY -> encode -> write .bin/.ply -> read .bin -> decode -> write decoded .ply -> verify
 *
 * Features kept:
 *   - Internal processing time (wall + user)
 *   - Reconstructed point cloud output
 *   - Payload buffer / bitstream size reporting
 * ============================================================================ */

#include "TMC3.h"

#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "PayloadBuffer.h"
#include "entropy.h"
#include "geometry.h"
#include "geometry_predictive.h"
#include "geometry_params.h"
#include "hls.h"
#include "io_hls.h"
#include "io_tlv.h"
#include "pcc_chrono.h"
#include "ply.h"
#include "PCCPointSet.h"
#include "version.h"

using namespace std;
using namespace pcc;

// ============================================================================
// Minimal parameters for the simplified flow
// ============================================================================
struct SimpleParams {
  string inputPlyPath;                         // --input
  string outputBitstreamPath = "predgeom.bin";  // --output
  string reconstructedPath = "recon.ply";       // --recon
  string decodedPath = "decoded.ply";           // --decoded
  double inputScale = 1.0;                      // --scale
  int    numGroups  = 1;                        // --groups
};

// ============================================================================
// Forward declaration — the simplified encodePredictiveGeometry in
// geometry_predictive_encoder.cpp (ring-based, no tree, no inter)
// ============================================================================
namespace pcc {
void encodePredictiveGeometry(
  const PredGeomEncOpts& opt,
  const GeometryParameterSet& gps,
  GeometryBrickHeader& gbh,
  PCCPointSet3& cloud,
  PredGeomContexts& ctxtMem,
  EntropyEncoder* arithmeticEncoder,
  int numGroups);

void decodePredictiveGeometry(
  const GeometryParameterSet& gps,
  const GeometryBrickHeader& gbh,
  PCCPointSet3& cloud,
  PredGeomContexts& ctxtMem,
  EntropyDecoder* arithmeticDecoder,
  int numGroups);
}

// ============================================================================
static bool
parseSimpleArgs(int argc, char* argv[], SimpleParams& p)
{
  if (argc < 3) {
    cerr << "Usage: " << argv[0]
         << " --input <ply> [--output <bitstream.bin>] [--recon <recon.ply>]"
         << " [--decoded <decoded.ply>]"
         << " [--scale <inputScale>] [--groups <1|2|4|8|16|32>]\n";
    return false;
  }
  for (int i = 1; i < argc; i++) {
    string arg = argv[i];
    if ((arg == "--input" || arg == "-i") && i + 1 < argc)
      p.inputPlyPath = argv[++i];
    else if ((arg == "--output" || arg == "-o") && i + 1 < argc)
      p.outputBitstreamPath = argv[++i];
    else if ((arg == "--recon" || arg == "-r") && i + 1 < argc)
      p.reconstructedPath = argv[++i];
    else if ((arg == "--decoded" || arg == "-d") && i + 1 < argc)
      p.decodedPath = argv[++i];
    else if ((arg == "--scale" || arg == "-s") && i + 1 < argc)
      p.inputScale = atof(argv[++i]);
    else if ((arg == "--groups" || arg == "-g") && i + 1 < argc)
      p.numGroups = atoi(argv[++i]);
    else {
      cerr << "Unknown option: " << arg << "\n";
      return false;
    }
  }
  if (p.inputPlyPath.empty()) {
    cerr << "Error: --input is required\n";
    return false;
  }
  return true;
}

// ============================================================================
//  Minimal defaults for the parameter sets used by PredGeomEncoder
// ============================================================================
static void
setupMinimalSPS(SequenceParameterSet& sps)
{
  sps.sps_seq_parameter_set_id = 0;
  sps.frame_ctr_bits = 16;
  sps.slice_tag_bits = 0;
  sps.geometry_axis_order = AxisOrder::kXYZ;
  sps.cabac_bypass_stream_enabled_flag = true;
  sps.bypass_bin_coding_without_prob_update = false;
  sps.entropy_continuation_enabled_flag = false;
  sps.inter_frame_prediction_enabled_flag = false;
  sps.inter_entropy_continuation_enabled_flag = false;

  sps.seqBoundingBoxOrigin = 0;
  sps.seqBoundingBoxSize  = 0;
  sps.sps_bounding_box_offset_bits = 0;
  sps.sps_bounding_box_size_bits = 0;
}

static void
setupMinimalGPS(GeometryParameterSet& gps)
{
  gps.gps_geom_parameter_set_id = 0;
  gps.gps_seq_parameter_set_id  = 0;
  gps.predgeom_enabled_flag = true;

  // No duplicate-point merging in our simplified flow
  gps.geom_unique_points_flag = false;

  // No angular / azimuth scaling — pure Cartesian residual
  gps.geom_angular_mode_enabled_flag = false;
  gps.azimuth_scaling_enabled_flag   = false;
  gps.geom_angular_azimuth_scale_log2_minus11 = 0;
  gps.geom_angular_azimuth_speed_minus1 = 0;
  gps.geom_angular_radius_inv_scale_log2 = 0;
  gps.geom_slice_angular_origin_present_flag = false;
  gps.gpsAngularOrigin = 0;

  // No geometry scaling / QP
  gps.geom_scaling_enabled_flag = false;
  gps.geom_qp_multiplier_log2  = 0;
  gps.geom_base_qp = 0;
  gps.geom_idcm_qp_offset = 0;
  gps.geom_qp_offset_intvl_log2 = 0;

  // Prediction list
  gps.predgeom_max_pred_index = 0;
  gps.predgeom_radius_threshold_for_pred_list = 0;
  gps.resR_context_qphi_threshold = 0;
  gps.resR_context_qphi_threshold_present_flag = false;

  // Residual2 disabled (not relevant for flat loop)
  gps.residual2_disabled_flag = true;

  // No trisoup / octree / qtbt / inter
  gps.trisoup_enabled_flag    = false;
  gps.qtbt_enabled_flag       = false;
  gps.interPredictionEnabledFlag = false;
  gps.globalMotionEnabled     = false;
  gps.biPredictionEnabledFlag = 0;

  // Laser table — we'll set numLasers = 0 (no angular table)
  gps.angularTheta.clear();
  gps.angularZ.clear();
  gps.angularNumPhiPerTurn.clear();

  // Other flags that PredGeomEncoder reads
  gps.geom_box_log2_scale_present_flag = false;
  gps.gps_geom_box_log2_scale = 0;
  gps.neighbour_avail_boundary_log2_minus1 = 0;
  gps.inferred_direct_coding_mode = 0;
  gps.geom_planar_mode_enabled_flag = false;
  gps.bitwise_occupancy_coding_flag = true;
  gps.octree_point_count_list_present_flag = false;
}

static void
setupMinimalGBH(GeometryBrickHeader& gbh)
{
  gbh.geom_geom_parameter_set_id = 0;
  gbh.geom_slice_id = 0;
  gbh.slice_tag = 0;
  gbh.frame_ctr_lsb = 0;
  gbh.prev_slice_id = 0;
  gbh.geomBoxOrigin = 0;
  gbh.gbhAngularOrigin = 0;
  gbh.geom_box_origin_bits_minus1 = 0;
  gbh.geom_box_log2_scale = 0;
  gbh.geom_slice_qp_offset = 0;
  gbh.geom_stream_cnt_minus1 = 0;       // single stream
  gbh.geom_qp_offset_intvl_log2_delta = 0;
  gbh.rootNodeSizeLog2 = 0;
  gbh.maxRootNodeDimLog2 = 0;
  gbh.entropy_continuation_flag = false;
  gbh.interPredictionEnabledFlag = false;
  gbh.biPredictionEnabledFlag = false;
  gbh.pgeom_min_radius = 0;

  // Residual bit-depth (set again inside encodePredictiveGeometry)
  for (int k = 0; k < 3; k++)
    gbh.pgeom_resid_abs_log2_bits[k] = 5;
}

static PredGeomEncOpts
makeMinimalPredGeomOpts()
{
  PredGeomEncOpts opt{};
  opt.sortMode = PredGeomEncOpts::kNoSort;
  opt.maxPtsPerTree = 0;
  opt.azimuthSortRecipBinWidth = 0;
  opt.maxPredIdxTested = 0;
  opt.radiusThresholdForNewPred = 0;
  opt.enablePartition = false;
  opt.splitter = 0;
  return opt;
}

static bool
isValidNumGroups(int g)
{
  return g == 1 || g == 2 || g == 4 || g == 8 || g == 16 || g == 32;
}

static bool
writeBitstreamFileTlv(const string& path, const PayloadBuffer& payload)
{
  ofstream fout(path, ios::binary);
  if (!fout.is_open())
    return false;
  writeTlv(payload, fout);
  return fout.good();
}

static bool
readBitstreamFileTlv(const string& path, PayloadBuffer& payload)
{
  ifstream fin(path, ios::binary);
  if (!fin.is_open())
    return false;
  readTlv(fin, &payload);
  if (!fin || payload.type != PayloadType::kGeometryBrick)
    return false;
  return true;
}

static bool
writePointCloudPly(const PCCPointSet3& cloud, const string& path)
{
  ply::PropertyNameMap names;
  names.position = {"x", "y", "z"};
  return ply::write(cloud, names, 0.001, {0, 0, 0}, path, /*ascii=*/true);
}

// ============================================================================
//  Main
// ============================================================================
int
main(int argc, char* argv[])
{
  cout << "=== TMC3 Simplified (PredGeom only) ===" << endl;

  // ---- 1. Parse options ----
  SimpleParams params;
  if (!parseSimpleArgs(argc, argv, params))
    return 1;
  if (!isValidNumGroups(params.numGroups)) {
    cerr << "Error: --groups must be one of {1,2,4,8,16,32}" << endl;
    return 1;
  }

  // ---- 2. Read PLY ----
  cout << "[1] Reading PLY: " << params.inputPlyPath << " ..." << endl;

  // Property names expected in the PLY (xyz + ring)
  ply::PropertyNameMap plyNames;
  plyNames.position = {"x", "y", "z"};

  PCCPointSet3 cloud;
  if (!ply::read(params.inputPlyPath, plyNames, params.inputScale, cloud)
      || cloud.getPointCount() == 0) {
    cerr << "Error: cannot open or empty input file!" << endl;
    return -1;
  }

  const auto numPoints = cloud.getPointCount();
  cout << "  Points loaded: " << numPoints
       << "  (inputScale=" << params.inputScale << ")" << endl;

  cout << "  Using implicit laser index cycle: 0..31 (ring field ignored)." << endl;

  // ---- 3. Set up minimal parameter sets ----
  SequenceParameterSet sps;
  setupMinimalSPS(sps);

  GeometryParameterSet gps;
  setupMinimalGPS(gps);

  GeometryBrickHeader gbh;
  setupMinimalGBH(gbh);

  PredGeomEncOpts predGeomOpt = makeMinimalPredGeomOpts();
  PredGeomContexts ctxtMem;

  // ---- 4. Encode ----
  // Wall + user timers
  pcc::chrono::Stopwatch<std::chrono::steady_clock> clock_wall;
  pcc::chrono::Stopwatch<pcc::chrono::utime_inc_children_clock> clock_user;

  cout << "\n[2] Encoding (PredictiveGeometry, flat ring-based loop, "
       << params.numGroups << " context group(s)) ..." << endl;

  // Start payload buffer
  PayloadBuffer payload(PayloadType::kGeometryBrick);

  // Allocate arithmetic encoder
  int maxAcBufLen = int(numPoints) * 3 * 4 + 1024;
  std::unique_ptr<EntropyEncoder> aec(new EntropyEncoder(maxAcBufLen, nullptr));
  aec->enableBypassStream(sps.cabac_bypass_stream_enabled_flag);
  aec->setBypassBinCodingWithoutProbUpdate(sps.bypass_bin_coding_without_prob_update);
  aec->start();

  // --- Timed section ---
  clock_wall.start();
  clock_user.start();

  encodePredictiveGeometry(
    predGeomOpt, gps, gbh, cloud,
    ctxtMem, aec.get(), params.numGroups);

  clock_user.stop();
  clock_wall.stop();

  // Signal the actual number of points coded
  gbh.footer.geom_num_points_minus1 = cloud.getPointCount() - 1;

  // ---- 5. Assemble payload buffer ----
  write(sps, gps, gbh, &payload);

  auto aecLen = aec->stop();
  auto aecBuf = aec->buffer();
  payload.insert(payload.end(), aecBuf, aecBuf + aecLen);
  write(gps, gbh, gbh.footer, &payload);

  // ---- 6. Write bitstream ----
  if (!writeBitstreamFileTlv(params.outputBitstreamPath, payload)) {
    cerr << "Error: could not write bitstream file: " << params.outputBitstreamPath << endl;
    return 1;
  }
  cout << "\n[3] Bitstream written: " << params.outputBitstreamPath
       << " (" << payload.size() << " B)" << endl;

  // ---- 7. Write encoder reconstruction ----
  if (!writePointCloudPly(cloud, params.reconstructedPath)) {
    cerr << "Error: could not write reconstructed PLY: " << params.reconstructedPath << endl;
    return 1;
  }
  cout << "[4] Encoder reconstruction written: " << params.reconstructedPath << endl;

  // ---- 8. Read bitstream file and decode ----
  cout << "\n[5] Decoding from bitstream file ..." << endl;

  PayloadBuffer payloadIn(PayloadType::kGeometryBrick);
  if (!readBitstreamFileTlv(params.outputBitstreamPath, payloadIn) || payloadIn.empty()) {
    cerr << "Error: failed to read bitstream file: " << params.outputBitstreamPath << endl;
    return 1;
  }

  int bytesReadHead = 0;
  int bytesReadFoot = 0;
  auto gbhFromBitstream = parseGbh(sps, gps, payloadIn, &bytesReadHead, &bytesReadFoot);
  if (bytesReadHead < 0 || bytesReadFoot < 0
      || bytesReadHead + bytesReadFoot > int(payloadIn.size())) {
    cerr << "Error: invalid payload layout while parsing geometry brick." << endl;
    return 1;
  }

  const int codedGeomLen = int(payloadIn.size()) - bytesReadHead - bytesReadFoot;
  if (codedGeomLen <= 0) {
    cerr << "Error: geometry arithmetic payload is empty." << endl;
    return 1;
  }

  PCCPointSet3 decCloud;
  PredGeomContexts decCtxtMem;
  std::unique_ptr<EntropyDecoder> aed(new EntropyDecoder());
  aed->setBuffer(codedGeomLen, payloadIn.data() + bytesReadHead);
  aed->enableBypassStream(sps.cabac_bypass_stream_enabled_flag);
  aed->setBypassBinCodingWithoutProbUpdate(sps.bypass_bin_coding_without_prob_update);
  aed->start();

  decodePredictiveGeometry(
    gps, gbhFromBitstream, decCloud,
    decCtxtMem, aed.get(), params.numGroups);

  // ---- 9. Write decoder output ----
  if (!writePointCloudPly(decCloud, params.decodedPath)) {
    cerr << "Error: could not write decoded PLY: " << params.decodedPath << endl;
    return 1;
  }
  cout << "[6] Decoded point cloud written: " << params.decodedPath << endl;

  // ---- 10. Verification ----
  cout << "\n[7] Verification ..." << endl;
  int mismatchCount = 0;
  if (decCloud.getPointCount() != cloud.getPointCount()) {
    cerr << "  Mismatch in point count! Encoded: " << cloud.getPointCount() 
         << ", Decoded: " << decCloud.getPointCount() << endl;
    mismatchCount++;
  } else {
    for (size_t i = 0; i < cloud.getPointCount(); i++) {
      if (cloud[i] != decCloud[i]) {
        if (mismatchCount < 10) {
          cerr << "  Mismatch at point " << i << ": "
               << "Enc=(" << cloud[i][0] << "," << cloud[i][1] << "," << cloud[i][2] << ") "
               << "Dec=(" << decCloud[i][0] << "," << decCloud[i][1] << "," << decCloud[i][2] << ")\n";
        }
        mismatchCount++;
      }
    }
  }

  if (mismatchCount == 0) {
    cout << "  Verification OK: all " << decCloud.getPointCount() << " reconstructed points match the encoded points!" << endl;
  } else {
    cout << "  Verification FAILED with " << mismatchCount << " mismatched points!" << endl;
  }

  // ---- 11. Report ----
  double bpp = double(8 * payload.size()) / numPoints;
  cout << "\n[8] Results:" << endl;
  cout << "  Payload size:     " << payload.size() << " B ("
       << bpp << " bpp)" << endl;

  using namespace std::chrono;
  auto total_wall = duration_cast<milliseconds>(clock_wall.count()).count();
  auto total_user = duration_cast<milliseconds>(clock_user.count()).count();
  cout << "  Processing time (wall): " << total_wall / 1000.0 << " s" << endl;
  cout << "  Processing time (user): " << total_user / 1000.0 << " s" << endl;

  cout << "\nDone." << endl;
  return 0;
}
