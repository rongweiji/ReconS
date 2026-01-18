#include "camera_info.hpp"
#include "depth_estimator.hpp"

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <iostream>
#include <set>
#include <string>
#include <vector>

namespace fs = std::filesystem;

struct Options {
  std::string rgb_dir;
  std::string out_dir;
  std::string engine_path;
  std::string calibration_path;
  double depth_scale = 1000.0;
  bool save_exr = false;
  bool skip_existing = false;
  int max_frames = -1;
  std::set<std::string> extensions;
  bool preview = false;
  std::string colormap = "JET";
  double depth_min = std::numeric_limits<double>::quiet_NaN();
  double depth_max = std::numeric_limits<double>::quiet_NaN();
};

static std::string default_engine_path()
{
  const fs::path root = fs::path(__FILE__).parent_path().parent_path();
  const fs::path model_dir = root / "models";
  const fs::path trt10 = model_dir / "DA3METRIC-LARGE.trt10.engine";
  const fs::path legacy = model_dir / "DA3METRIC-LARGE.fp16-batch1.engine";
  if (fs::exists(trt10)) {
    return trt10.string();
  }
  return legacy.string();
}

static void print_usage(const char * prog)
{
  std::cout << "Usage: " << prog << " --rgb-dir DIR [options]\n"
            << "  --engine PATH        TensorRT engine file\n"
            << "  --calibration PATH   Calibration YAML (camera_info or K1/D1/R1/P1)\n"
            << "  --camera-info PATH   Alias for --calibration\n"
            << "  --out-dir DIR        Output directory (default: sibling <rgb_dir>_depth)\n"
            << "  --depth-scale VALUE  Scale meters to uint16 (default 1000.0)\n"
            << "  --save-exr           Save float32 depth .exr instead of 16-bit PNG\n"
            << "  --skip-existing      Skip frames with existing outputs\n"
            << "  --max-frames N        Process only first N frames\n"
            << "  --ext LIST            Comma-separated extensions (default jpg,jpeg,png,bmp,tif,tiff)\n"
            << "  --preview            Show RGB + depth preview while processing\n"
            << "  -h, --help           Show this message\n";
}

static std::set<std::string> default_extensions()
{
  return {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"};
}

static std::string normalize_ext(const std::string & ext)
{
  std::string out;
  out.reserve(ext.size() + 1);
  if (ext.empty()) {
    return out;
  }
  if (ext[0] != '.') {
    out.push_back('.');
  }
  for (const char ch : ext) {
    out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
  }
  return out;
}

static std::set<std::string> parse_extensions(const std::string & value)
{
  std::set<std::string> exts;
  std::string token;
  for (char ch : value) {
    if (ch == ',') {
      if (!token.empty()) {
        exts.insert(normalize_ext(token));
        token.clear();
      }
    } else {
      token.push_back(ch);
    }
  }
  if (!token.empty()) {
    exts.insert(normalize_ext(token));
  }
  return exts;
}

static Options parse_args(int argc, char ** argv)
{
  Options opt;
  opt.engine_path = default_engine_path();
  opt.extensions = default_extensions();

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error("Missing value for " + arg);
      }
      return argv[++i];
    };

    if (arg == "--rgb-dir") {
      opt.rgb_dir = next();
    } else if (arg == "--out-dir") {
      opt.out_dir = next();
    } else if (arg == "--engine") {
      opt.engine_path = next();
    } else if (arg == "--calibration") {
      opt.calibration_path = next();
    } else if (arg == "--camera-info") {
      opt.calibration_path = next();
    } else if (arg == "--depth-scale") {
      opt.depth_scale = std::stod(next());
    } else if (arg == "--save-exr") {
      opt.save_exr = true;
    } else if (arg == "--skip-existing") {
      opt.skip_existing = true;
    } else if (arg == "--max-frames") {
      opt.max_frames = std::stoi(next());
    } else if (arg == "--ext") {
      opt.extensions = parse_extensions(next());
    } else if (arg == "--preview") {
      opt.preview = true;
    } else if (arg == "-h" || arg == "--help") {
      print_usage(argv[0]);
      std::exit(0);
    } else {
      throw std::runtime_error("Unknown option: " + arg);
    }
  }

  if (opt.rgb_dir.empty() || opt.out_dir.empty()) {
    if (opt.rgb_dir.empty()) {
      throw std::runtime_error("--rgb-dir is required");
    }
  }

  if (opt.calibration_path.empty()) {
    throw std::runtime_error("--calibration is required");
  }

  if (opt.out_dir.empty()) {
    const fs::path rgb_path(opt.rgb_dir);
    const std::string base = rgb_path.filename().string() + "_depth";
    opt.out_dir = (rgb_path.parent_path() / base).string();
  }

  return opt;
}

static bool has_extension(const fs::path & path, const std::set<std::string> & exts)
{
  if (exts.empty()) {
    return true;
  }
  const std::string ext = normalize_ext(path.extension().string());
  return exts.find(ext) != exts.end();
}

static std::vector<fs::path> collect_images(const fs::path & dir, const std::set<std::string> & exts)
{
  std::vector<fs::path> files;
  if (!fs::exists(dir) || !fs::is_directory(dir)) {
    return files;
  }
  for (const auto & entry : fs::directory_iterator(dir)) {
    if (!entry.is_regular_file()) {
      continue;
    }
    const fs::path path = entry.path();
    if (!has_extension(path, exts)) {
      continue;
    }
    files.push_back(path);
  }
  std::sort(files.begin(), files.end());
  return files;
}

static int colormap_code(const std::string & name)
{
  if (name == "HOT") return cv::COLORMAP_HOT;
  if (name == "COOL") return cv::COLORMAP_COOL;
  if (name == "BONE") return cv::COLORMAP_BONE;
  if (name == "VIRIDIS") return cv::COLORMAP_VIRIDIS;
  if (name == "PLASMA") return cv::COLORMAP_PLASMA;
  if (name == "INFERNO") return cv::COLORMAP_INFERNO;
  if (name == "MAGMA") return cv::COLORMAP_MAGMA;
  return cv::COLORMAP_JET;
}

static cv::Mat depth_to_colormap(const cv::Mat & depth, const Options & opt)
{
  if (depth.empty()) {
    return cv::Mat();
  }

  cv::Mat depth_float;
  if (depth.type() == CV_32F) {
    depth_float = depth;
  } else {
    depth.convertTo(depth_float, CV_32F);
  }

  cv::Mat mask = depth_float > 0.0f;
  double min_val = 0.0;
  double max_val = 0.0;

  const bool has_min = std::isfinite(opt.depth_min);
  const bool has_max = std::isfinite(opt.depth_max);
  if (has_min) {
    min_val = opt.depth_min;
  }
  if (has_max) {
    max_val = opt.depth_max;
  }
  if (!has_min || !has_max) {
    double auto_min = 0.0;
    double auto_max = 0.0;
    cv::minMaxLoc(depth_float, &auto_min, &auto_max, nullptr, nullptr, mask);
    if (!has_min) {
      min_val = auto_min;
    }
    if (!has_max) {
      max_val = auto_max;
    }
  }

  if (max_val <= min_val) {
    max_val = min_val + 1.0;
  }

  cv::Mat normalized;
  depth_float.convertTo(normalized, CV_32F, 1.0 / (max_val - min_val), -min_val / (max_val - min_val));
  cv::Mat clipped;
  cv::min(cv::max(normalized, 0.0f), 1.0f, clipped);

  cv::Mat gray;
  clipped.convertTo(gray, CV_8U, 255.0);

  cv::Mat colored;
  cv::applyColorMap(gray, colored, colormap_code(opt.colormap));
  return colored;
}

int main(int argc, char ** argv)
{
  Options opt;
  try {
    opt = parse_args(argc, argv);
  } catch (const std::exception & e) {
    std::cerr << e.what() << "\n";
    print_usage(argv[0]);
    return 1;
  }

  const fs::path rgb_dir = fs::path(opt.rgb_dir);
  const fs::path out_dir = fs::path(opt.out_dir);
  fs::create_directories(out_dir);

  const std::vector<fs::path> images = collect_images(rgb_dir, opt.extensions);
  if (images.empty()) {
    std::cerr << "No images found in " << rgb_dir << "\n";
    return 1;
  }

  sensor_msgs::msg::CameraInfo cam_info;
  const bool calib_loaded = xlernav::LoadCameraInfoYaml(opt.calibration_path, cam_info);
  if (!calib_loaded) {
    std::cerr << "Warning: failed to load camera info from " << opt.calibration_path << "\n";
  }

  xlernav::DepthEstimator depth_engine(opt.engine_path);

  if (opt.preview) {
    cv::namedWindow("RGB + Depth", cv::WINDOW_NORMAL);
  }

  int processed = 0;
  for (const auto & image_path : images) {
    if (opt.max_frames > 0 && processed >= opt.max_frames) {
      break;
    }

    const std::string filename = image_path.filename().string();
    fs::path out_path = out_dir / image_path.stem();
    out_path.replace_extension(opt.save_exr ? ".exr" : ".png");

    if (opt.skip_existing && fs::exists(out_path)) {
      ++processed;
      continue;
    }

    cv::Mat bgr = cv::imread(image_path.string(), cv::IMREAD_COLOR);
    if (bgr.empty()) {
      std::cerr << "Failed to read " << image_path << "\n";
      ++processed;
      continue;
    }

    if (cam_info.width == 0 || cam_info.height == 0) {
      cam_info.width = static_cast<uint32_t>(bgr.cols);
      cam_info.height = static_cast<uint32_t>(bgr.rows);
    }

    if (!depth_engine.Infer(bgr, cam_info)) {
      std::cerr << "Depth inference failed for " << image_path << "\n";
      ++processed;
      continue;
    }

    cv::Mat depth = depth_engine.Depth();
    if (depth.empty()) {
      std::cerr << "Empty depth result for " << image_path << "\n";
      ++processed;
      continue;
    }

    if (opt.save_exr) {
      cv::Mat depth_float;
      if (depth.type() == CV_32F) {
        depth_float = depth;
      } else {
        depth.convertTo(depth_float, CV_32F);
      }
      if (!cv::imwrite(out_path.string(), depth_float)) {
        std::cerr << "Failed to write " << out_path << "\n";
      }
    } else {
      cv::Mat depth_float;
      if (depth.type() == CV_32F) {
        depth_float = depth;
      } else {
        depth.convertTo(depth_float, CV_32F);
      }
      cv::Mat valid = depth_float > 0.0f;
      cv::Mat depth_scaled;
      depth_float.convertTo(depth_scaled, CV_32F, opt.depth_scale);
      depth_scaled.setTo(0.0f, ~valid);
      cv::threshold(depth_scaled, depth_scaled, 65535.0f, 65535.0f, cv::THRESH_TRUNC);
      cv::Mat depth_u16;
      depth_scaled.convertTo(depth_u16, CV_16U);
      if (!cv::imwrite(out_path.string(), depth_u16)) {
        std::cerr << "Failed to write " << out_path << "\n";
      }
    }

    if (opt.preview) {
      cv::Mat depth_vis = depth_to_colormap(depth, opt);
      if (depth_vis.empty()) {
        depth_vis = cv::Mat(bgr.rows, bgr.cols, CV_8UC3, cv::Scalar(0, 0, 0));
      }
      cv::Mat combined;
      cv::hconcat(bgr, depth_vis, combined);
      cv::imshow("RGB + Depth", combined);
      const int key = cv::waitKey(1) & 0xFF;
      if (key == 27 || key == 'q') {
        std::cout << "Preview stopped by user.\n";
        break;
      }
    }

    ++processed;
    if (processed % 10 == 0) {
      std::cout << "Processed " << processed << " / " << images.size() << "\n";
    }
  }

  if (opt.preview) {
    cv::destroyAllWindows();
  }

  std::cout << "Done. Processed " << processed << " frames.\n";
  return 0;
}
