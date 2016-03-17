#include "G4chroma.hh"
#include "GLG4Scint.hh"
#include <G4SteppingManager.hh>
#include <G4OpticalPhysics.hh>
#include <G4EmPenelopePhysics.hh>
#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"

#include <iostream>

using namespace std;

ChromaPhysicsList::ChromaPhysicsList():  G4VModularPhysicsList()
{
  // default cut value  (1.0mm) 
  defaultCutValue = 1.0*mm;

  // General Physics
  RegisterPhysics( new G4EmPenelopePhysics(0) );
  // Optical Physics w/o Scintillation
  G4OpticalPhysics* opticalPhysics = new G4OpticalPhysics();
  opticalPhysics->Configure(kScintillation,false);
  RegisterPhysics( opticalPhysics );
  // Scintillation (handled by stepping!)
  new GLG4Scint();
}

ChromaPhysicsList::~ChromaPhysicsList()
{
}

void ChromaPhysicsList::SetCuts(){
  //  " G4VUserPhysicsList::SetCutsWithDefault" method sets 
  //   the default cut value for all particle types 
  SetCutsWithDefault();   
}

SteppingAction::SteppingAction()
{
    scint = true;
}

SteppingAction::~SteppingAction()
{
}

void SteppingAction::EnableScint(bool enabled) {
    scint = enabled;
    cout << "Set scintillation to: " << enabled << endl;
}

void SteppingAction::UserSteppingAction(const G4Step* aStep) {

    if (scint) {
        
        G4VParticleChange * pParticleChange = GLG4Scint::GenericPostPostStepDoIt(aStep);
        if (!pParticleChange) return;
             
        const size_t nsecondaries = pParticleChange->GetNumberOfSecondaries();
        for (size_t i = 0; i < nsecondaries; i++) { 
            G4Track * tempSecondaryTrack = pParticleChange->GetSecondary(i);
            fpSteppingManager->GetfSecondary()->push_back( tempSecondaryTrack );
        }
        
        pParticleChange->Clear();
        
    }
    
}


TrackingAction::TrackingAction()
{
}

TrackingAction::~TrackingAction()
{
}

int TrackingAction::GetNumPhotons() const
{
  return pos.size();
}

void TrackingAction::Clear()
{
  pos.clear();
  dir.clear();
  pol.clear();
  wavelength.clear();
  t0.clear();
}

void TrackingAction::GetX(double *x) const
{
  for (unsigned i=0; i < pos.size(); i++) x[i] = pos[i].x();
}

void TrackingAction::GetY(double *y) const
{
  for (unsigned i=0; i < pos.size(); i++) y[i] = pos[i].y();
}

void TrackingAction::GetZ(double *z) const
{
  for (unsigned i=0; i < pos.size(); i++) z[i] = pos[i].z();
}

void TrackingAction::GetDirX(double *dir_x) const
{
  for (unsigned i=0; i < dir.size(); i++) dir_x[i] = dir[i].x();
}

void TrackingAction::GetDirY(double *dir_y) const
{
  for (unsigned i=0; i < dir.size(); i++) dir_y[i] = dir[i].y();
}

void TrackingAction::GetDirZ(double *dir_z) const
{
  for (unsigned i=0; i < dir.size(); i++) dir_z[i] = dir[i].z();
}

void TrackingAction::GetPolX(double *pol_x) const
{
  for (unsigned i=0; i < pol.size(); i++) pol_x[i] = pol[i].x();
}

void TrackingAction::GetPolY(double *pol_y) const
{
  for (unsigned i=0; i < pol.size(); i++) pol_y[i] = pol[i].y();
}

void TrackingAction::GetPolZ(double *pol_z) const
{
  for (unsigned i=0; i < pol.size(); i++) pol_z[i] = pol[i].z();
}

void TrackingAction::GetWavelength(double *wl) const
{
  for (unsigned i=0; i < wavelength.size(); i++) wl[i] = wavelength[i];
}

void TrackingAction::GetT0(double *t) const
{
  for (unsigned i=0; i < t0.size(); i++) t[i] = t0[i];
}

void TrackingAction::PreUserTrackingAction(const G4Track *track)
{
    G4ParticleDefinition *particle = track->GetDefinition();
    if (particle->GetParticleName() == "opticalphoton") {
        pos.push_back(track->GetPosition()/mm);
        dir.push_back(track->GetMomentumDirection());
        pol.push_back(track->GetPolarization());
        wavelength.push_back( (h_Planck * c_light / track->GetKineticEnergy()) / nanometer );
        t0.push_back(track->GetGlobalTime() / ns);
        const_cast<G4Track *>(track)->SetTrackStatus(fStopAndKill);
    }
}

#include <boost/python.hpp>
#include <pyublas/numpy.hpp>

using namespace boost::python;

pyublas::numpy_vector<double> PTA_GetX(const TrackingAction *pta)
{
  pyublas::numpy_vector<double> r(pta->GetNumPhotons());
  pta->GetX(&r[0]);
  return r;
}

pyublas::numpy_vector<double> PTA_GetY(const TrackingAction *pta)
{
  pyublas::numpy_vector<double> r(pta->GetNumPhotons());
  pta->GetY(&r[0]);
  return r;
}

pyublas::numpy_vector<double> PTA_GetZ(const TrackingAction *pta)
{
  pyublas::numpy_vector<double> r(pta->GetNumPhotons());
  pta->GetZ(&r[0]);
  return r;
}

pyublas::numpy_vector<double> PTA_GetDirX(const TrackingAction *pta)
{
  pyublas::numpy_vector<double> r(pta->GetNumPhotons());
  pta->GetDirX(&r[0]);
  return r;
}

pyublas::numpy_vector<double> PTA_GetDirY(const TrackingAction *pta)
{
  pyublas::numpy_vector<double> r(pta->GetNumPhotons());
  pta->GetDirY(&r[0]);
  return r;
}

pyublas::numpy_vector<double> PTA_GetDirZ(const TrackingAction *pta)
{
  pyublas::numpy_vector<double> r(pta->GetNumPhotons());
  pta->GetDirZ(&r[0]);
  return r;
}

pyublas::numpy_vector<double> PTA_GetPolX(const TrackingAction *pta)
{
  pyublas::numpy_vector<double> r(pta->GetNumPhotons());
  pta->GetPolX(&r[0]);
  return r;
}

pyublas::numpy_vector<double> PTA_GetPolY(const TrackingAction *pta)
{
  pyublas::numpy_vector<double> r(pta->GetNumPhotons());
  pta->GetPolY(&r[0]);
  return r;
}

pyublas::numpy_vector<double> PTA_GetPolZ(const TrackingAction *pta)
{
  pyublas::numpy_vector<double> r(pta->GetNumPhotons());
  pta->GetPolZ(&r[0]);
  return r;
}

pyublas::numpy_vector<double> PTA_GetWave(const TrackingAction *pta)
{
  pyublas::numpy_vector<double> r(pta->GetNumPhotons());
  pta->GetWavelength(&r[0]);
  return r;
}

pyublas::numpy_vector<double> PTA_GetT0(const TrackingAction *pta)
{
  pyublas::numpy_vector<double> r(pta->GetNumPhotons());
  pta->GetT0(&r[0]);
  return r;
}


void export_Chroma()
{
  class_<ChromaPhysicsList, ChromaPhysicsList*, bases<G4VModularPhysicsList>, boost::noncopyable > ("ChromaPhysicsList", "EM+Optics physics list")
    .def(init<>())
    ;

  class_<SteppingAction, SteppingAction*, bases<G4UserSteppingAction>,
	 boost::noncopyable > ("SteppingAction", "Stepping action for hacking purposes")
    .def(init<>())
    .def("EnableScint",&SteppingAction::EnableScint)
    ;  
  
  class_<TrackingAction, TrackingAction*, bases<G4UserTrackingAction>,
	 boost::noncopyable > ("TrackingAction", "Tracking action that saves photons")
    .def(init<>())
    .def("GetNumPhotons", &TrackingAction::GetNumPhotons)
    .def("Clear", &TrackingAction::Clear)
    .def("GetX", PTA_GetX)
    .def("GetY", PTA_GetY)
    .def("GetZ", PTA_GetZ)
    .def("GetDirX", PTA_GetDirX)
    .def("GetDirY", PTA_GetDirY)
    .def("GetDirZ", PTA_GetDirZ)
    .def("GetPolX", PTA_GetPolX)
    .def("GetPolY", PTA_GetPolY)
    .def("GetPolZ", PTA_GetPolZ)
    .def("GetWavelength", PTA_GetWave)
    .def("GetT0", PTA_GetT0)
    ;
}

BOOST_PYTHON_MODULE(_g4chroma)
{
  export_Chroma();
}
