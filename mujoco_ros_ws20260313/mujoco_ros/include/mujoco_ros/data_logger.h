#ifndef JUMP_TRAJECTORY_PLANNING_MPC_CPP_CPP_DATA_LOGGER_H
#define JUMP_TRAJECTORY_PLANNING_MPC_CPP_CPP_DATA_LOGGER_H

#include <iostream>
#include <fstream>
#include <string>
#include <ctime>

class DataLogger {
public:
  void init(const std::string &_file_name_prefix, const bool &_enable_time_suffix,
            const std::string *_header_str_array, const int &_cols);
  void log(const double *_data_array);
  void close();
  static void getCurrentTimeString(char *time_str_p, int str_len);
  DataLogger & operator<< (const double &_data) {
    ++count_;
    fout_ << _data;
    return *this;
  }
  DataLogger & operator, (const double &_data) {
    ++count_;
    fout_ << ", " << _data;
    return *this;
  }
  DataLogger & operator<< (std::ostream& (*op) (std::ostream&)) {
    if (count_ != table_cols_) {
      throw std::string("DataLogger:: The cols of data should equal to the cols of header.");
    }
    count_ = 0;
    fout_ << std::endl;
//    op(fout_);
    return *this;
  }

private:
  std::string log_filename;
  std::ofstream fout_;
  int table_cols_;
  int count_;
};


#endif //JUMP_TRAJECTORY_PLANNING_MPC_CPP_CPP_DATA_LOGGER_H
