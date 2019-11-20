#include <TVector3.h>
#include <vector>
#include <TTree.h>
#include <string>
#include <map>

struct Vertex {
  virtual ~Vertex() { };

  std::string particle_name;
  TVector3 pos;
  TVector3 dir;
  TVector3 pol;
  double ke;
  double t0;
  int trackid;
  int pdgcode;
  
  std::vector<Vertex> children;
  std::vector<double> x,y,z,t,px,py,pz,child_ke,edep;

  ClassDef(Vertex, 1);
};

struct Photon {
  virtual ~Photon() { };

  double t;
  TVector3 pos;
  TVector3 dir;
  TVector3 pol;
  double wavelength; // nm
  unsigned int flag;
  int last_hit_triangle;

  ClassDef(Photon, 1);
};

struct Channel {
  Channel() : id(-1), t(-1e9), q(-1e9) { };
  virtual ~Channel() { };

  int id;
  double t;
  double q;
  unsigned int flag;

  ClassDef(Channel, 1);
};

struct Event {
  virtual ~Event() { };

  int id;
  unsigned int nhit;
  unsigned int nchannels;
  
  double TotalQ() const {
    double sum = 0.0;
    for (unsigned int i=0; i < channels.size(); i++)
      sum += channels[i].q;
    return sum;
  }


  std::vector<Vertex> vertices;
  std::vector<Photon> photons_beg;
  std::vector<Photon> photons_end;
  std::map<int,Photon> hits;
  std::vector<Channel> channels;

  ClassDef(Event, 1);
};

void fill_channels(Event *ev, unsigned int nhit, unsigned int *ids, float *t,
		   float *q, unsigned int *flags, unsigned int nchannels)
{
  ev->nhit = 0;
  ev->nchannels = nchannels;
  ev->channels.resize(0);

  Channel ch;
  unsigned int id;
  for (unsigned int i=0; i < nhit; i++) {
      ev->nhit++;
      id = ids[i];
      ch.id = id;
      ch.t = t[id];
      ch.q = q[id];
      ch.flag = flags[id];
      ev->channels.push_back(ch);
  }
}

void get_channels(Event *ev, int *hit, float *t, float *q, unsigned int *flags)
{
  for (unsigned int i=0; i < ev->nchannels; i++) {
    hit[i] = 0;
    t[i] = -1e9f;
    q[i] = -1e9f;
    flags[i] = 0;
  }

  unsigned int id;
  for (unsigned int i=0; i < ev->channels.size(); i++) {
    id = ev->channels[i].id;

    if (id < ev->nchannels) {
      hit[id] = 1;
      t[id] = ev->channels[i].t;
      q[id] = ev->channels[i].q;
      flags[id] = ev->channels[i].flag;
    }
  }
}

void get_photons(const std::vector<Photon> &photons, float *pos, float *dir,
		 float *pol, float *wavelengths, float *t,
		 int *last_hit_triangles, unsigned int *flags)
{
  for (unsigned int i=0; i < photons.size(); i++) {
    const Photon &photon = photons[i];
    pos[3*i] = photon.pos.X();
    pos[3*i+1] = photon.pos.Y();
    pos[3*i+2] = photon.pos.Z();

    dir[3*i] = photon.dir.X();
    dir[3*i+1] = photon.dir.Y();
    dir[3*i+2] = photon.dir.Z();
    
    pol[3*i] = photon.pol.X();
    pol[3*i+1] = photon.pol.Y();
    pol[3*i+2] = photon.pol.Z();

    wavelengths[i] = photon.wavelength;
    t[i] = photon.t;
    flags[i] = photon.flag;
    last_hit_triangles[i] = photon.last_hit_triangle;
  }
}
		 
void fill_photons(std::vector<Photon> &photons,
		  unsigned int nphotons, float *pos, float *dir,
		  float *pol, float *wavelengths, float *t,
		  int *last_hit_triangles, unsigned int *flags)
{
  photons.resize(nphotons);
  
  for (unsigned int i=0; i < nphotons; i++) {
    Photon &photon = photons[i];
    photon.t = t[i];
    photon.pos.SetXYZ(pos[3*i], pos[3*i + 1], pos[3*i + 2]);
    photon.dir.SetXYZ(dir[3*i], dir[3*i + 1], dir[3*i + 2]);
    photon.pol.SetXYZ(pol[3*i], pol[3*i + 1], pol[3*i + 2]);
    photon.wavelength = wavelengths[i];
    photon.last_hit_triangle = last_hit_triangles[i];
    photon.flag = flags[i];

  }
}

#ifdef __MAKECINT__
#pragma link C++ class vector<Vertex>;
#pragma link C++ class vector<Photon>;
#pragma link C++ class vector<Channel>;
#endif


