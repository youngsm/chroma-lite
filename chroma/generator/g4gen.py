from chroma.generator.mute import *

import pyublas
import numpy as np
from chroma.event import Photons, Vertex, Steps
from chroma.tools import argsort_direction

#g4mute()
from Geant4 import *
#g4unmute()
import g4py.ezgeom
import g4py.NISTmaterials
import g4py.ParticleGun
from chroma.generator import _g4chroma

class G4Generator(object):
    def __init__(self, material, seed=None):
        """Create generator to produce photons inside the specified material.

           material: chroma.geometry.Material object with density, 
                     composition dict and refractive_index.

                     composition dictionary should be 
                        { element_symbol : fraction_by_weight, ... }.

           seed: int, *optional*
               Random number generator seed for HepRandom. If None, generator
               is not seeded.
        """
        if seed is not None:
            HepRandom.setTheSeed(seed)

        g4py.NISTmaterials.Construct()
        self.world_material = self.create_g4material(material)
        g4py.ezgeom.Construct()
        self.physics_list = _g4chroma.ChromaPhysicsList()
        gRunManager.SetUserInitialization(self.physics_list)
        self.particle_gun = g4py.ParticleGun.Construct()
        
        g4py.ezgeom.SetWorldMaterial(self.world_material)
        g4py.ezgeom.ResizeWorld(100*m, 100*m, 100*m)

        self.world = g4py.ezgeom.G4EzVolume('world')
        self.world.CreateBoxVolume(self.world_material, 100*m, 100*m, 100*m)
        self.world.PlaceIt(G4ThreeVector(0,0,0))

        self.stepping_action = _g4chroma.SteppingAction()
        gRunManager.SetUserAction(self.stepping_action)
        self.tracking_action = _g4chroma.TrackingAction()
        gRunManager.SetUserAction(self.tracking_action)
        #g4mute()
        gRunManager.Initialize()
        #g4unmute()
        # preinitialize the process by running a simple event
        self.generate_photons([Vertex('e-', (0,0,0), (1,0,0), 0.5, 1.0)], mute=True)
        
    def create_g4material(self, material):
        g4material = G4Material('world_material', material.density * g / cm3,
                                len(material.composition))

        # Add elements
        for element_name, element_frac_by_weight in material.composition.items():
            g4material.AddElement(G4Element.GetElement(element_name, True),
                                  element_frac_by_weight)

        # Set index of refraction
        prop_table = G4MaterialPropertiesTable()
        # Reverse entries so they are in ascending energy order rather
        # than wavelength
        energy = list((2*pi*hbarc / (material.refractive_index[::-1,0] * nanometer)).astype(float))
        values = list(material.refractive_index[::-1, 1].astype(float))
        prop_table.AddProperty('RINDEX', energy, values)
        if material.scintillation_light_yield:
            prop_table.AddConstProperty('LIGHT_YIELD',material.scintillation_light_yield)
        if np.any(material.scintillation_spectrum):
            energy = list((2*pi*hbarc / (material.scintillation_spectrum[::-1,0] * nanometer)).astype(float))
            values = list(material.scintillation_spectrum[::-1, 1].astype(float))
            prop_table.AddProperty('SCINTILLATION', energy, values)
        if np.any(material.scintillation_spectrum):
            energy = list(material.scintillation_waveform[:, 0].astype(float))
            values = list(material.scintillation_waveform[:, 1].astype(float))
            prop_table.AddProperty('SCINTWAVEFORM', energy, values)
        if np.any(material.scintillation_mod):
            energy = list(material.scintillation_mod[:, 0].astype(float))
            values = list(material.scintillation_mod[:, 1].astype(float))
            prop_table.AddProperty('SCINTMOD', energy, values)

        # Load properties
        g4material.SetMaterialPropertiesTable(prop_table)
        return g4material

    def _extract_photons_from_tracking_action(self, sort=True):
        n = self.tracking_action.GetNumPhotons()        
        pos = np.zeros(shape=(n,3), dtype=np.float32)
        pos[:,0] = self.tracking_action.GetX()
        pos[:,1] = self.tracking_action.GetY()
        pos[:,2] = self.tracking_action.GetZ()

        dir = np.zeros(shape=(n,3), dtype=np.float32)
        dir[:,0] = self.tracking_action.GetDirX()
        dir[:,1] = self.tracking_action.GetDirY()
        dir[:,2] = self.tracking_action.GetDirZ()

        pol = np.zeros(shape=(n,3), dtype=np.float32)
        pol[:,0] = self.tracking_action.GetPolX()
        pol[:,1] = self.tracking_action.GetPolY()
        pol[:,2] = self.tracking_action.GetPolZ()
        
        wavelengths = self.tracking_action.GetWavelength().astype(np.float32)

        t0 = self.tracking_action.GetT0().astype(np.float32)

        if sort:
            reorder = argsort_direction(dir)
            pos = pos[reorder]
            dir = dir[reorder]
            pol = pol[reorder]
            wavelengths = wavelengths[reorder]
            t0 = t0[reorder]

        return Photons(pos, dir, pol, wavelengths, t0)
    
    def _extract_vertex_from_stepping_action(self, index=1, tracks=True):
        track = self.stepping_action.getTrack(index)
        steps = Steps(track.getStepX(),track.getStepY(),track.getStepZ(),track.getStepT(),
                      track.getStepPX(),track.getStepPY(),track.getStepPZ(),track.getStepKE(),
                      track.getStepEDep())
        children = [self._extract_vertex_from_stepping_action(track.getChildTrackID(id),steps) for id in xrange(track.getNumChildren())]
        return Vertex(track.name, np.array([steps.x[0],steps.y[0],steps.z[0]]), 
                        np.array([steps.px[0],steps.py[0],steps.pz[0]]), 
                        steps.ke[0], steps.t[0], steps=steps, children=children, trackid=index, pdgcode=track.pdg_code)
        

    def generate_photons(self, vertices, mute=False):
        """Use GEANT4 to generate photons produced by propagating `vertices`.
           
        Args:
            vertices: list of event.Vertex objects
                List of initial particle vertices.

            mute: bool
                Disable GEANT4 output to console during generation.  (GEANT4 can
                be quite chatty.)

        Returns:
            photons: event.Photons
                Photon vertices generated by the propagation of `vertices`.
        """
        if mute:
            pass
            #g4mute()

        photons = None

        try:
            tracked_vertices = []
            for vertex in vertices:
                self.particle_gun.SetParticleByName(vertex.particle_name)
                #FIXME validate this - Geant4 seems to call 'ParticleEnergy' KineticEnergy
                #mass = G4ParticleTable.GetParticleTable().FindParticle(vertex.particle_name).GetPDGMass()
                total_energy = vertex.ke*MeV# + mass
                self.particle_gun.SetParticleEnergy(total_energy)

                # Must be float type to call GEANT4 code
                pos = np.asarray(vertex.pos, dtype=np.float64)
                dir = np.asarray(vertex.dir, dtype=np.float64)

                self.particle_gun.SetParticlePosition(G4ThreeVector(*pos)*mm)
                self.particle_gun.SetParticleMomentumDirection(G4ThreeVector(*dir).unit())
                self.particle_gun.SetParticleTime(vertex.t0*ns)

                if vertex.pol is not None:
                    self.particle_gun.SetParticlePolarization(G4ThreeVector(*vertex.pol).unit())

                self.tracking_action.Clear()
                self.stepping_action.clearTracking()
                gRunManager.BeamOn(1)
                
                tracked_vertices.append(self._extract_vertex_from_stepping_action())

                if photons is None:
                    photons = self._extract_photons_from_tracking_action()
                else:
                    photons += self._extract_photons_from_tracking_action()
                
        finally:
            if mute:
                pass
                #g4unmute()

        return (tracked_vertices,photons)
