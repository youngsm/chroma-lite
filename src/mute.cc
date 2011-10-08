#include <geant4/G4ios.hh>

class discard_streambuf : public std::streambuf {
public:
  discard_streambuf() { };
  
  virtual int_type overflow(int_type c) {
    // Do nothing with this character
    return c;
  };
};

discard_streambuf discard;
std::streambuf *g4cout_orig = G4cout.rdbuf();
std::streambuf *g4cerr_orig = G4cerr.rdbuf();

void mute_g4mute() {
  G4cout.rdbuf(&discard);
  G4cerr.rdbuf(&discard);
}

void mute_g4unmute() {
  G4cout.rdbuf(g4cout_orig);
  G4cerr.rdbuf(g4cerr_orig);
}


#include <boost/python.hpp>

using namespace boost::python;

void export_mute()
{
  def("g4mute", mute_g4mute, default_call_policies(), "Silence all GEANT4 output");
  def("g4unmute", mute_g4unmute, default_call_policies(), "Re-enable GEANT4 output after calling ``g4mute()``.");
}

BOOST_PYTHON_MODULE(mute)
{
  export_mute();
}
