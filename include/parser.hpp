#pragma once

/* args_parse Parser class:
 *
 * Initialise with an optional program name Parser parser(<PROG_NAME>)
 *
 * Add any options / args required using addOption and addPositional.
 * .addOption<TYPE>(<LONG_NAME>, <CHAR>, <HELP_DESC>,
 * .addPositional(<NAME>).
 * If not a simple switch (true/false) use:
 * .addOption<TYPE>(<LONG_NAME>, <CHAR>, <HELP_DESC>,
 *                  <DEFAULT_VAL>, <NUM_REQ_ARGS>)
 *
 * Parse input using .parse(argc, argv).
 *
 * Options can be accessed using [<NAME>] (returns true if used)
 * or .option<TYPE>(<NAME>) that will return the value casted as <TYPE>.
 * Positional accessed using .positional(<NAME>).
 *
 * Passing the parser object to a stream will print generated help text
 * Additional text to printed can be added using .addText(<TEXT>).
 */

#include <map>
#include <string>
#include <vector>

class Arg {
private:
  int _nargs;
  std::string _desc;
  std::string _default;

public:
  Arg(const int nargs, const std::string &desc = "",
      const std::string &default_value = "")
      : _nargs(nargs), _desc(desc), _default(default_value) {}
};

class Parser {
private:
  std::string _prog;
  std::string _;
  std::map<std::string, Arg> _args;

public:
  template <typename T>
  void addArg(const std::string &name, const int nargs = 0,
              const std::string &desc = "",
              const std::string &default_value = "") {
    auto arg = Arg(nargs, desc, default_value);
    _args.insert({name, arg});
  }

  void parse(const std::string &text) {}
};
