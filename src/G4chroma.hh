#ifndef __G4chroma_hh__
#define __G4chroma_hh__

#include <G4VModularPhysicsList.hh>
class ChromaPhysicsList: public G4VModularPhysicsList
{
public:
  ChromaPhysicsList();
  virtual ~ChromaPhysicsList();
  virtual void SetCuts();
};

#include <G4UserSteppingAction.hh>
#include <G4UserTrackingAction.hh>
#include <vector>
#include <G4ThreeVector.hh>

class SteppingAction : public G4UserSteppingAction
{
public:
  SteppingAction();
  virtual ~SteppingAction();
  
  void EnableScint(bool enabled);
  
  void UserSteppingAction(const G4Step* aStep);
  
private:
    bool scint;

};

class TrackingAction : public G4UserTrackingAction
{
public:
  TrackingAction();
  virtual ~TrackingAction();
  
  int GetNumPhotons() const;
  void Clear();
  
  void GetX(double *x) const;
  void GetY(double *y) const;
  void GetZ(double *z) const;
  void GetDirX(double *dir_x) const;
  void GetDirY(double *dir_y) const;
  void GetDirZ(double *dir_z) const;
  void GetPolX(double *pol_x) const;
  void GetPolY(double *pol_y) const;
  void GetPolZ(double *pol_z) const;

  void GetWavelength(double *wl) const;
  void GetT0(double *t) const;

  virtual void PreUserTrackingAction(const G4Track *);

protected:
  std::vector<G4ThreeVector> pos;
  std::vector<G4ThreeVector> dir;
  std::vector<G4ThreeVector> pol;
  std::vector<double> wavelength;
  std::vector<double> t0;
};

#endif
