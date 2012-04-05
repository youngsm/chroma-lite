#include "G4chroma.hh"
#include <G4OpticalPhysics.hh>
#include <G4EmPenelopePhysics.hh>

ChromaPhysicsList::ChromaPhysicsList():  G4VModularPhysicsList()
{
  // default cut value  (1.0mm) 
  defaultCutValue = 1.0*mm;

  // General Physics
  RegisterPhysics( new G4EmPenelopePhysics(0) );
  // Optical Physics
  G4OpticalPhysics* opticalPhysics = new G4OpticalPhysics();
  RegisterPhysics( opticalPhysics );
}

ChromaPhysicsList::~ChromaPhysicsList()
{
}

void ChromaPhysicsList::SetCuts(){
  //  " G4VUserPhysicsList::SetCutsWithDefault" method sets 
  //   the default cut value for all particle types 
  SetCutsWithDefault();   
}


PhotonTrackingAction::PhotonTrackingAction()
{
}

PhotonTrackingAction::~PhotonTrackingAction()
{
}

int PhotonTrackingAction::GetNumPhotons() const
{
  return pos.size();
}

void PhotonTrackingAction::Clear()
{
  pos.clear();
  dir.clear();
  pol.clear();
  wavelength.clear();
  t0.clear();
}

void PhotonTrackingAction::GetX(double *x) const
{
  for (unsigned i=0; i < pos.size(); i++) x[i] = pos[i].x();
}

void PhotonTrackingAction::GetY(double *y) const
{
  for (unsigned i=0; i < pos.size(); i++) y[i] = pos[i].y();
}

void PhotonTrackingAction::GetZ(double *z) const
{
  for (unsigned i=0; i < pos.size(); i++) z[i] = pos[i].z();
}

void PhotonTrackingAction::GetDirX(double *dir_x) const
{
  for (unsigned i=0; i < dir.size(); i++) dir_x[i] = dir[i].x();
}

void PhotonTrackingAction::GetDirY(double *dir_y) const
{
  for (unsigned i=0; i < dir.size(); i++) dir_y[i] = dir[i].y();
}

void PhotonTrackingAction::GetDirZ(double *dir_z) const
{
  for (unsigned i=0; i < dir.size(); i++) dir_z[i] = dir[i].z();
}

void PhotonTrackingAction::GetPolX(double *pol_x) const
{
  for (unsigned i=0; i < pol.size(); i++) pol_x[i] = pol[i].x();
}

void PhotonTrackingAction::GetPolY(double *pol_y) const
{
  for (unsigned i=0; i < pol.size(); i++) pol_y[i] = pol[i].y();
}

void PhotonTrackingAction::GetPolZ(double *pol_z) const
{
  for (unsigned i=0; i < pol.size(); i++) pol_z[i] = pol[i].z();
}

void PhotonTrackingAction::GetWavelength(double *wl) const
{
  for (unsigned i=0; i < wavelength.size(); i++) wl[i] = wavelength[i];
}

void PhotonTrackingAction::GetT0(double *t) const
{
  for (unsigned i=0; i < t0.size(); i++) t[i] = t0[i];
}

void PhotonTrackingAction::PreUserTrackingAction(const G4Track *track)
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

pyublas::numpy_vector<double> PTA_GetX(const PhotonTrackingAction *pta)
{
  pyublas::numpy_vector<double> r(pta->GetNumPhotons());
  pta->GetX(&r[0]);
  return r;
}

pyublas::numpy_vector<double> PTA_GetY(const PhotonTrackingAction *pta)
{
  pyublas::numpy_vector<double> r(pta->GetNumPhotons());
  pta->GetY(&r[0]);
  return r;
}

pyublas::numpy_vector<double> PTA_GetZ(const PhotonTrackingAction *pta)
{
  pyublas::numpy_vector<double> r(pta->GetNumPhotons());
  pta->GetZ(&r[0]);
  return r;
}

pyublas::numpy_vector<double> PTA_GetDirX(const PhotonTrackingAction *pta)
{
  pyublas::numpy_vector<double> r(pta->GetNumPhotons());
  pta->GetDirX(&r[0]);
  return r;
}

pyublas::numpy_vector<double> PTA_GetDirY(const PhotonTrackingAction *pta)
{
  pyublas::numpy_vector<double> r(pta->GetNumPhotons());
  pta->GetDirY(&r[0]);
  return r;
}

pyublas::numpy_vector<double> PTA_GetDirZ(const PhotonTrackingAction *pta)
{
  pyublas::numpy_vector<double> r(pta->GetNumPhotons());
  pta->GetDirZ(&r[0]);
  return r;
}

pyublas::numpy_vector<double> PTA_GetPolX(const PhotonTrackingAction *pta)
{
  pyublas::numpy_vector<double> r(pta->GetNumPhotons());
  pta->GetPolX(&r[0]);
  return r;
}

pyublas::numpy_vector<double> PTA_GetPolY(const PhotonTrackingAction *pta)
{
  pyublas::numpy_vector<double> r(pta->GetNumPhotons());
  pta->GetPolY(&r[0]);
  return r;
}

pyublas::numpy_vector<double> PTA_GetPolZ(const PhotonTrackingAction *pta)
{
  pyublas::numpy_vector<double> r(pta->GetNumPhotons());
  pta->GetPolZ(&r[0]);
  return r;
}

pyublas::numpy_vector<double> PTA_GetWave(const PhotonTrackingAction *pta)
{
  pyublas::numpy_vector<double> r(pta->GetNumPhotons());
  pta->GetWavelength(&r[0]);
  return r;
}

pyublas::numpy_vector<double> PTA_GetT0(const PhotonTrackingAction *pta)
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

  class_<PhotonTrackingAction, PhotonTrackingAction*, bases<G4UserTrackingAction>,
	 boost::noncopyable > ("PhotonTrackingAction", "Tracking action that saves photons")
    .def(init<>())
    .def("GetNumPhotons", &PhotonTrackingAction::GetNumPhotons)
    .def("Clear", &PhotonTrackingAction::Clear)
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
