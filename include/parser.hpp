#pragma once

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

class ArgumentParser {
private:
  std::vector<std::string> args;
  bool valid = true;
  std::vector<std::string> help_strings;

public:
  ArgumentParser(int argc, char *argv[]) {

    args = std::vector<std::string>(argv, argv + argc);
  }

  template <typename T>
  std::string help_string(const std::string &name, const T &default_value,
                          bool required, const std::string &help) {
    std::ostringstream oss;
    oss << "--" << name;
    if (required)
      oss << " (required)";
    oss << ", " << help << ", default = " << default_value;
    return oss.str();
  }

  template <typename T>
  T read(const std::string &name, const T &default_value,
         const std::string &help = "", bool required = false) {

    help_strings.push_back(help_string(name, default_value, required, help));

    T value = default_value;
    for (auto it = args.begin(); it != args.end(); ++it) {
      if (*it == "--help")
        valid = false;
      if ((*it).substr(0, 2) == "--" && (*it).substr(2) == name) {
        // shortcut for flags
        if constexpr (std::is_same<T, bool>::value) {
          return true;
        }
        std::istringstream iss(*(++it));
        if (!(iss >> value)) {
          std::cerr << "unable to read '" + *it + "' into arg '" + name + "'"
                    << std::endl;
          valid = false;
        };
        return value;
      }
    } // end for

    if (required) {
      std::cerr << "missing required argument '" + name + "'" << std::endl;
      valid = false;
    }
    return value;
  }

  bool success() { return valid; }

  friend std::ostream &operator<<(std::ostream &os, const ArgumentParser &p) {
    for (auto it = p.help_strings.begin(); it != p.help_strings.end(); ++it) {
      os << *it << std::endl;
    }
    return os;
  }
};
