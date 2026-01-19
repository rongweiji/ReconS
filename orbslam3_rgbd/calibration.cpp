#include "calibration.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>

namespace {

std::string Trim(const std::string & input)
{
  const auto start = input.find_first_not_of(" \t\r\n");
  if (start == std::string::npos) {
    return "";
  }
  const auto end = input.find_last_not_of(" \t\r\n");
  return input.substr(start, end - start + 1);
}

bool StartsWith(const std::string & text, const std::string & prefix)
{
  return text.size() >= prefix.size() && text.compare(0, prefix.size(), prefix) == 0;
}

void ExtractNumbers(const std::string & line, std::vector<double> & values)
{
  std::string token;
  bool has_digit = false;

  auto flush = [&]() {
    if (has_digit) {
      try {
        values.push_back(std::stod(token));
      } catch (...) {
      }
    }
    token.clear();
    has_digit = false;
  };

  for (const char ch : line) {
    if (std::isdigit(static_cast<unsigned char>(ch)) || ch == '-' || ch == '+' || ch == '.' ||
        ch == 'e' || ch == 'E') {
      if (std::isdigit(static_cast<unsigned char>(ch))) {
        has_digit = true;
      }
      token.push_back(ch);
    } else if (!token.empty()) {
      flush();
    }
  }
  if (!token.empty()) {
    flush();
  }
}

}  // namespace

namespace recons {

bool LoadCalibrationYaml(const std::string & path, CalibrationData & calib)
{
  std::ifstream file(path);
  if (!file.is_open()) {
    return false;
  }

  enum class Section { kNone, kK, kD, kR, kP };
  Section section = Section::kNone;
  bool in_data = false;
  bool in_list = false;
  bool in_image_size = false;

  std::vector<double> k_data;
  std::vector<double> d_data;
  std::vector<double> r_data;
  std::vector<double> p_data;

  std::string line;
  while (std::getline(file, line)) {
    const std::string trimmed = Trim(line);
    if (trimmed.empty()) {
      continue;
    }

    if (StartsWith(trimmed, "image_width:")) {
      const auto value = Trim(trimmed.substr(std::string("image_width:").size()));
      try {
        calib.width = std::stoi(value);
      } catch (...) {
      }
      continue;
    }
    if (StartsWith(trimmed, "image_height:")) {
      const auto value = Trim(trimmed.substr(std::string("image_height:").size()));
      try {
        calib.height = std::stoi(value);
      } catch (...) {
      }
      continue;
    }
    if (StartsWith(trimmed, "image_size:")) {
      in_image_size = true;
      continue;
    }
    if (in_image_size) {
      std::vector<double> size_values;
      ExtractNumbers(trimmed, size_values);
      if (!size_values.empty()) {
        if (calib.width <= 0 && size_values.size() >= 1) {
          calib.width = static_cast<int>(size_values[0]);
        }
        if (calib.height <= 0 && size_values.size() >= 2) {
          calib.height = static_cast<int>(size_values[1]);
        }
      }
      if (calib.width > 0 && calib.height > 0) {
        in_image_size = false;
      }
      continue;
    }

    if (StartsWith(trimmed, "distortion_model:")) {
      calib.distortion_model = Trim(trimmed.substr(std::string("distortion_model:").size()));
      if (!calib.distortion_model.empty() && calib.distortion_model.front() == '"' &&
          calib.distortion_model.back() == '"') {
        calib.distortion_model = calib.distortion_model.substr(1, calib.distortion_model.size() - 2);
      }
      continue;
    }

    if (trimmed == "camera_matrix:") {
      section = Section::kK;
      in_data = false;
      in_list = false;
      continue;
    }
    if (trimmed == "distortion_coefficients:") {
      section = Section::kD;
      in_data = false;
      in_list = false;
      continue;
    }
    if (trimmed == "rectification_matrix:") {
      section = Section::kR;
      in_data = false;
      in_list = false;
      continue;
    }
    if (trimmed == "projection_matrix:") {
      section = Section::kP;
      in_data = false;
      in_list = false;
      continue;
    }
    if (trimmed == "K:" || trimmed == "K1:") {
      section = Section::kK;
      in_list = true;
      in_data = false;
      continue;
    }
    if (trimmed == "D:" || trimmed == "D1:") {
      section = Section::kD;
      in_list = true;
      in_data = false;
      continue;
    }
    if (trimmed == "R:" || trimmed == "R1:") {
      section = Section::kR;
      in_list = true;
      in_data = false;
      continue;
    }
    if (trimmed == "P:" || trimmed == "P1:") {
      section = Section::kP;
      in_list = true;
      in_data = false;
      continue;
    }

    if (StartsWith(trimmed, "data:")) {
      in_data = true;
      continue;
    }

    if (!in_data && !in_list) {
      continue;
    }

    std::vector<double> * target = nullptr;
    switch (section) {
      case Section::kK:
        target = &k_data;
        break;
      case Section::kD:
        target = &d_data;
        break;
      case Section::kR:
        target = &r_data;
        break;
      case Section::kP:
        target = &p_data;
        break;
      default:
        break;
    }
    if (!target) {
      continue;
    }
    ExtractNumbers(trimmed, *target);
  }

  if (k_data.size() >= 9) {
    calib.k.assign(k_data.begin(), k_data.begin() + 9);
  }
  if (!d_data.empty()) {
    calib.d = d_data;
  }
  if (r_data.size() >= 9) {
    calib.r.assign(r_data.begin(), r_data.begin() + 9);
  }
  if (p_data.size() >= 12) {
    calib.p.assign(p_data.begin(), p_data.begin() + 12);
  }

  if (calib.distortion_model.empty()) {
    calib.distortion_model = "plumb_bob";
  }

  return calib.k.size() >= 9;
}

bool BuildUndistortMaps(
  const CalibrationData & calib,
  double balance,
  bool use_projection,
  const std::string & distortion_override,
  cv::Mat & map1,
  cv::Mat & map2,
  cv::Mat & new_k)
{
  if (calib.k.size() < 9) {
    return false;
  }

  const int width = calib.width > 0 ? calib.width : 640;
  const int height = calib.height > 0 ? calib.height : 480;

  cv::Mat k = (cv::Mat_<double>(3, 3) <<
    calib.k[0], calib.k[1], calib.k[2],
    calib.k[3], calib.k[4], calib.k[5],
    calib.k[6], calib.k[7], calib.k[8]);

  const std::string model = distortion_override.empty() ? calib.distortion_model : distortion_override;
  const bool use_fisheye = (model == "equidistant" || model == "fisheye");

  const size_t d_count = use_fisheye ? 4u : 5u;
  cv::Mat d = cv::Mat::zeros(static_cast<int>(d_count), 1, CV_64F);
  if (!calib.d.empty()) {
    const size_t count = std::min(d_count, calib.d.size());
    for (size_t i = 0; i < count; ++i) {
      d.at<double>(static_cast<int>(i), 0) = calib.d[i];
    }
  }

  cv::Mat r = cv::Mat::eye(3, 3, CV_64F);
  if (calib.r.size() >= 9) {
    r = (cv::Mat_<double>(3, 3) <<
      calib.r[0], calib.r[1], calib.r[2],
      calib.r[3], calib.r[4], calib.r[5],
      calib.r[6], calib.r[7], calib.r[8]);
  }

  const cv::Size size(width, height);
  const double clamped_balance = std::clamp(balance, 0.0, 1.0);

  const bool has_projection = calib.p.size() >= 12;
  const bool use_projection_matrix = use_projection && has_projection;
  if (use_projection_matrix) {
    new_k = (cv::Mat_<double>(3, 3) <<
      calib.p[0], calib.p[1], calib.p[2],
      calib.p[4], calib.p[5], calib.p[6],
      calib.p[8], calib.p[9], calib.p[10]);
  }

  if (use_fisheye) {
    if (!use_projection_matrix) {
      cv::fisheye::estimateNewCameraMatrixForUndistortRectify(
        k, d, size, r, new_k, clamped_balance);
    }
    cv::fisheye::initUndistortRectifyMap(
      k, d, r, new_k, size, CV_16SC2, map1, map2);
  } else {
    if (!use_projection_matrix) {
      new_k = cv::getOptimalNewCameraMatrix(k, d, size, clamped_balance);
    }
    cv::initUndistortRectifyMap(k, d, r, new_k, size, CV_16SC2, map1, map2);
  }

  return !map1.empty();
}

}  // namespace recons
