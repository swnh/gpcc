/* ============================================================================
 * TMC3_simple.cpp
 *
 * Simplified TMC3 flow:
 *   option setting → PLY read → encodeGeometryBrick → encodePredictiveGeometry
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
  string inputPlyPath;         // --input
  string outputBitstreamPath;  // --output  (raw payload, optional)
  string reconstructedPath;    // --recon   (reconstructed PLY, optional)
  double inputScale = 1.0;     // --scale   (position scale for PLY read)
  int    numGroups  = 1;       // --groups  (ring context groups: 1,2,4,8,16,32)
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
  const int* ring,
  int numGroups);
}

// ============================================================================
static bool
parseSimpleArgs(int argc, char* argv[], SimpleParams& p)
{
  if (argc < 3) {
    cerr << "Usage: " << argv[0]
         << " --input <ply> [--output <bitstream>] [--recon <ply>]"
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
  sps.cabac_bypass_stream_enabled_flag = false;
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

  // Extract ring array from the cloud (loaded as "laserAngle" / "ring")
  vector<int> ringVec(numPoints, 0);
  if (cloud.hasLaserAngles()) {
    for (size_t i = 0; i < numPoints; i++)
      ringVec[i] = cloud.getLaserAngle(i);
    cout << "  Ring/laser data found in PLY." << endl;
  } else {
    cout << "  WARNING: no ring/laser data in PLY — using ring=0 for all points." << endl;
  }

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
    ctxtMem, aec.get(), ringVec.data(), params.numGroups);

  clock_user.stop();
  clock_wall.stop();

  // Signal the actual number of points coded
  gbh.footer.geom_num_points_minus1 = cloud.getPointCount() - 1;

  // ---- 5. Assemble payload buffer ----
  // Write GBH header into payload
  write(sps, gps, gbh, &payload);

  // Flush AEC and append to payload
  auto aecLen = aec->stop();
  auto aecBuf = aec->buffer();
  payload.insert(payload.end(), aecBuf, aecBuf + aecLen);

  // Append footer
  write(gps, gbh, gbh.footer, &payload);

  // ---- 6. Report ----
  double bpp = double(8 * payload.size()) / numPoints;
  cout << "\n[3] Results:" << endl;
  cout << "  Payload size:     " << payload.size() << " B ("
       << bpp << " bpp)" << endl;

  using namespace std::chrono;
  auto total_wall = duration_cast<milliseconds>(clock_wall.count()).count();
  auto total_user = duration_cast<milliseconds>(clock_user.count()).count();
  cout << "  Processing time (wall): " << total_wall / 1000.0 << " s" << endl;
  cout << "  Processing time (user): " << total_user / 1000.0 << " s" << endl;

  // ---- 7. Write bitstream (optional) ----
  if (!params.outputBitstreamPath.empty()) {
    ofstream fout(params.outputBitstreamPath, ios::binary);
    if (fout.is_open()) {
      fout.write(payload.data(), payload.size());
      fout.close();
      cout << "  Bitstream written to: " << params.outputBitstreamPath << endl;
    } else {
      cerr << "  Warning: could not open output file: "
           << params.outputBitstreamPath << endl;
    }
  }

  // ---- 8. Write reconstructed PLY (optional) ----
  // After encodePredictiveGeometry, `cloud` has been reordered in coded order.
  // The reconstruction debug PLY is also written from inside the function.
  if (!params.reconstructedPath.empty()) {
    ply::PropertyNameMap reconNames;
    reconNames.position = {"x", "y", "z"};
    if (ply::write(cloud, reconNames, 0.001, {0, 0, 0},
                   params.reconstructedPath, /*ascii=*/true)) {
      cout << "  Reconstructed PLY: " << params.reconstructedPath << endl;
    } else {
      cerr << "  Warning: could not write reconstructed PLY." << endl;
    }
  }

  cout << "\nDone." << endl;
  return 0;
}
