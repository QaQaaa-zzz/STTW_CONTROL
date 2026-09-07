#include "data_logger.h"

void DataLogger::init(const std::string &_file_name_prefix, const bool &_enable_time_suffix,
                      const std::string *_header_str_array, const int &_cols) {
  // 文件保存初始化
  log_filename = ".txt";
  if (_enable_time_suffix) {
    char cur_time_str[16];
    getCurrentTimeString(cur_time_str, 16);
    log_filename.insert(0,cur_time_str);
    log_filename.insert(0,"_");
  }
  log_filename.insert(0,_file_name_prefix);
  fout_.open(log_filename);     //创建一个data.txt的文件
  std::cout << "DataLogger:: " << log_filename << std::endl;

  table_cols_ = _cols;
  // 记录数据表头
  for (int i = 0; i < table_cols_; ++i) {
    fout_ << *(_header_str_array + i);
    if (i == table_cols_ - 1) {
      fout_ << std::endl;
    } else {
      fout_ << ", ";
    }
  }
  count_ = 0;
}

void DataLogger::log(const double *_data_array) {
  // 记录数据
  for (int i = 0; i < table_cols_; ++i) {
    fout_ << *(_data_array + i);
    if (i == table_cols_ - 1) {
      fout_ << std::endl;
    } else {
      fout_ << ", ";
    }
  }
}

void DataLogger::close() {
  fout_.close();
}

void DataLogger::getCurrentTimeString(char *time_str_p, int str_len) {
  time_t timep;
  time(&timep);
  strftime(time_str_p, str_len, "%Y%m%d-%H%M%S", localtime(&timep));
}