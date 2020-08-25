import numpy as np
import dolfin as dl
import pathlib, os
from pyapprox.fenics_models.fenics_utilities import generate_polygonal_mesh, get_polygon_boundary_segments
from pyapprox.fenics_models.helmholtz import run_model
from pyapprox.benchmarks.benchmarks import Benchmark

def generate_helmholtz_bases(samples,mesh_resolution=51):
    frequency=400;speaker_amplitudes=1
    sound_speed=np.array([343,6320,343])
    kappa,forcing,function_space,boundary_conditions,nedges,nsegments_per_edge,bndry_obj=\
        setup_helmholtz_model(
            mesh_resolution,frequency,speaker_amplitudes,
            sound_speed)

    speaker_segment_indices, cabinet_segment_indices = \
        get_speaker_boundary_segment_indices(nedges,nsegments_per_edge)

    sols = []
    for jj in speaker_segment_indices:
        boundary_conditions =get_helmholtz_speaker_boundary_conditions(nedges,nsegments_per_edge,bndry_obj,speaker_amplitudes,frequency)
        for kk in range(nedges*nsegments_per_edge):
            if jj!=kk:
                boundary_conditions[kk][2][1]=dl.Constant(0)
        pii=run_model(kappa,forcing,function_space,boundary_conditions)
        sols.append(pii)

    basis = np.empty((samples.shape[1],len(sols)),dtype=float)
    for ii in range(samples.shape[1]):
        for jj in range(len(sols)):
            basis[ii,jj] = sols[jj](samples[0,ii],samples[1,ii])[1]

    return basis, sols, function_space

def helmholtz_basis(sols,active_speakers,samples):
    sols = [sols[ii] for ii in active_speakers]
    basis = np.empty((samples.shape[1],len(sols)),dtype=float)
    for ii in range(samples.shape[1]):
        for jj in range(len(sols)):
            basis[ii,jj] = sols[jj](samples[0,ii],samples[1,ii])[1]
    return basis

class HelmholtzBasis(object):
    def __init__(self,sols,active_speakers):
        self.sols=sols
        self.active_speakers = np.array(active_speakers)
        
    def __call__(self,samples):
        return helmholtz_basis(self.sols,self.active_speakers,samples)

    def get_indices(self):
        return self.active_speakers.copy()[np.newaxis,:]

def get_helmholtz_speaker_boundary_conditions(nedges,nsegments_per_edge,bndry_obj,speaker_amplitudes,frequency):
    speaker_segment_indices, cabinet_segment_indices = \
        get_speaker_boundary_segment_indices(nedges,nsegments_per_edge)
    if np.isscalar(speaker_amplitudes):
        speaker_amplitudes = np.array([speaker_amplitudes]*len(speaker_segment_indices))

    #1.204 is the standard atmospheric pressure
    omega = 2.*np.pi*frequency
    boundary_conditions=[
        ['neumann',bndry_obj[ii],
         [dl.Constant(0),dl.Constant(1.204*omega*speaker_amplitudes[jj])]]
        for jj,ii in enumerate(speaker_segment_indices)]

    #gamma=8.4e-4
    #alpha=kappa*dl.Constant(gamma)
    #make solution entirely imaginary, i.e. real part is zero
    alpha=0
    if alpha==0:
        #print('neumann')
        boundary_conditions+=[
            ['neumann',bndry_obj[ii], [dl.Constant(0),dl.Constant(0)]]
            for ii in cabinet_segment_indices]
    else:
        boundary_conditions+=[
            ['robin',bndry_obj[ii],
             [dl.Constant(0),dl.Constant(0)],[dl.Constant(0),alpha]]
            for ii in cabinet_segment_indices]

    tmp = [None for ii in range(len(boundary_conditions))]
    for ii in range(len(speaker_segment_indices)):
        tmp[speaker_segment_indices[ii]]=boundary_conditions[ii]
    for ii in range(len(cabinet_segment_indices)):
        tmp[cabinet_segment_indices[ii]]=\
            boundary_conditions[ii+len(speaker_segment_indices)]
    boundary_conditions=tmp
    return boundary_conditions

def get_speaker_boundary_segment_indices(nedges,nsegments_per_edge):
    assert nsegments_per_edge>2
    speaker_segment_indices = []
    for ii in range(nedges):
        speaker_segment_indices+=list(np.arange(ii*nsegments_per_edge,(ii+1)*nsegments_per_edge)[1:-1])
    cabinet_segment_indices = np.setdiff1d(np.arange(nedges*nsegments_per_edge),speaker_segment_indices)
    #print(speaker_segment_indices, cabinet_segment_indices)
    return speaker_segment_indices, cabinet_segment_indices

def setup_helmholtz_model(mesh_resolution,frequency=400,speaker_amplitudes=1,
                          sound_speed=np.array([343,6320,343])):
    """
    # 6320 speed of sound in aluminium
    # 343 speed of sound of air at 20 degrees celsius
    """
    nedges=8
    ampothem=1.5
    radius=0.5
    #edge_length = 2*np.tan(180/nedges)
    #print(edge_length)

    # mesh creation uses random ray tracing so load mesh from file
    mesh_filename = 'helmholtz-mesh-res-%d.xml'%mesh_resolution
    if not os.path.exists(mesh_filename):
        #print('generating mesh')
        mesh = generate_polygonal_mesh(
            mesh_resolution,ampothem,nedges,radius,False)
        mesh_file = dl.File(mesh_filename)
        mesh_file << mesh
    else:
        #print('loading mesh', mesh_filename)
        mesh=dl.Mesh(mesh_filename)

    degree=1
    P1=dl.FiniteElement('Lagrange',mesh.ufl_cell(),degree)
    element=dl.MixedElement([P1,P1])
    function_space=dl.FunctionSpace(mesh,element)

    omega = 2.*np.pi*frequency

    kappas = omega/sound_speed
    cx1 = 0; cx2 = cx1-radius/np.sqrt(8)
    kappa = dl.Expression('((x[0]-c1)*(x[0]-c1)+(x[1]-c1)*(x[1]-c1) >= r*r + tol) ? k_0 : ((x[0]-c2)*(x[0]-c2)+(x[1]-c2)*(x[1]-c2)>=r*r/4+tol ? k_1 : k_2)', degree=0, tol=1e-14, k_0=kappas[0], k_1=kappas[1],k_2=kappas[2],r=radius,c1=cx1,c2=cx2)

    #pp=dl.plot(kappa,mesh=mesh)
    #plt.colorbar(pp)
    #plt.show()

    forcing=[dl.Constant(0),dl.Constant(0)]

    nsegments_per_edge = 3
    cumulative_segment_sizes = [0.0625,0.9375,1.]
    bndry_obj = get_polygon_boundary_segments(ampothem,nedges,nsegments_per_edge)
    boundary_conditions = get_helmholtz_speaker_boundary_conditions(nedges,nsegments_per_edge,bndry_obj,speaker_amplitudes,frequency)
    return kappa,forcing,function_space,boundary_conditions,nedges,\
        nsegments_per_edge, bndry_obj

def setup_mfnets_helmholtz_benchmark(mesh_resolution=51,noise_std=1):
    r"""
    Setup the multi-fidelity benchmark used to combine helmholtz data.

    .. figure:: ../figures/helmholtz-octagon.png
       :align: center

       Experimental configuration. Speaker cones (green speakers), speaker cabinets (black segments), scatterer (red and blue circles).

    For a fixed angular velocity :math:`\omega = 2\pi f`, the acoustic pressure :math:`u` is modeled using the (real) Helmholtz equation defined on an open regular octagon domain :math:`D` with apothem equal to 1.5 meters. The interior of :math:`D` contains a scatterer (red and blue circles) and each side of the octagon consists of an individual speaker and its cabinet; the centered green boundary segments are speaker cones which comprise 0.875 of the total edge length and the black segments are the cabinet walls. To simplify the problem, we model the scatterer as a dense fluid and ignore the impedance of the speaker cabinet.

    .. math::  \Delta u + \kappa^2 u = 0 \quad\text{in }D, \qquad\qquad\frac{\partial u}{\partial n} = \rho_0\omega\sum_{j=1}^8 \theta_j\chi_j \quad\text{on }\partial D

    where :math:`\kappa=\omega/c` is the wave number, :math:`c` is the speed of sound, :math:`\rho_0` is the fluid density, :math:`\chi_j:\partial D\to\{0,1\}` is the characteristic function of the :math:`j^{\text{th}}` speaker cone (green boundary segments in the figure), and :math:`\theta_j` is the acoustic velocity output by the :math:`j^{\text{th}}` speaker for :math:`j=1,\ldots,8` --- in other words, the :math:`j^{\text{th}}` speaker cone oscillates with velocity :math:`\theta_j\cos(\omega t)`. In this example we assume that the material in the red circle is made of aluminum for which the speed of sound is 6320 m/s and that the regions in the blue circle and exterior to the red circle are comprised of air at :math:`20^\circ\text{C}` which has a speed of sound of 343 m/s.  In addition, we set the frequency to be :math:`f=400` Hz and the fluid density to be that of air at :math:`20^\circ\text{C}` and standard atmospheric pressure, i.e. :math:`\rho_0=1.204 \text{kg/m}^3`.

    The benchmark consists of data from three experiments each consisting of measurements of the acoustic pressure data :math:`u(x)`, at 5000 microphone locations :math:`x`. For the high-fideliy experiment we set the speaker amplitudes as :math:`\theta_{1,i}=1`, :math:`i=1,\ldots,8` and for the low-fidelity experiments we set :math:`\theta_{2,i}=1`, :math:`i=3,5,7` and :math:`\theta_{3,i}=3`, :math:`i=2,4,6,8`; all other speaker amplitudes are set to zero. Speakers are ordered counter clockwise with the first speaker located on the right vertical edge of the octagon. 

    Under these conditions each information source

    .. math:: 

        u_1(x)=\sum_{i=1}^8 \phi_i(x)\theta_{1,i} \qquad   u_2(x)=\sum_{i=3,5,7} \phi_i(x)\theta_{2,i} \qquad   u_3(x)=\sum_{i=2,4,6,8} \phi_i(x)\theta_{3,i}

    is a linear sum of basis functions :math:`\phi_i(x)` which correspond to solving the Helmholtz equation using only one active speaker. Specifically the basis :math:`\phi_i` is obtained by solving

    .. math::

       \Delta \phi + \kappa^2 \phi = 0 \quad\text{in }D,
       \qquad\qquad\frac{\partial \phi}{\partial n} =  
       \rho_0\omega\theta_i \quad\text{on }\partial D

    The measurements of each experiment, taken at locations :math:`x_k^{(i)}` in the domain :math:`D`, are given by :math:`y_k^{(i)}=u_k(x_k^{(i)})+\epsilon_k^{(i)}` for each information source :math:`k=1,2,3`, where the noise :math:`\epsilon_k^{(i)}` is normally distributed with mean zero and unit variance. 


    Parameters
    ----------
    mesh_resolution : integer
        Specifies the resolution of the finite element mesh

    noise_std : float
        The standard deviation of the IID noise in the observations

    Returns
    -------
    benchmark : pya.Benchmark
        Object containing the benchmark attributes documented below

    samples : list 
        List with entries np.ndarray (nvars,nsamples) which contain the 
        locations :math:`x_k^{(i)}` of the observations of each 
        information source

    obs : list np.ndarray (nsamples,1)
        List with entries np.ndarray (nsamples,1) containing the values of
        the noisy observations :math:`y_k^{(i)}` of each information source

    bases : np.ndarray (nsamples,8)
        The basis functions :math:`\phi_i` used to model the observations
        
    noise_std : float
        The standard deviation of the IID noise in the observations. 
        This is just the value passed in to this function, stored for 
        convienience.
    """
    active_speakers = [[0,1,2,3,4,5,6,7],[2,4,6],[1,3,5,7]]
    amplitudes = [[1,1,1,1,1,1,1,1],[2,2,2],[3,3,3,3]]
    nsources = len(active_speakers)

    data_path = pathlib.Path(__file__).parent.absolute()
    filename = os.path.join(data_path,'helmholtz_noise_data.txt')
    samples=[np.loadtxt(filename)[:,:2].T.copy() for ii in range(nsources)]

    basis,sols,function_space=generate_helmholtz_bases(
        samples[0],mesh_resolution)
    
    values = [basis[:,active_speakers[ii]].dot(amplitudes[ii])[:,np.newaxis]
              for ii in range(nsources)]
    
    if np.isscalar(noise_std):
        noise_std = [noise_std]*nsources
    np.random.seed(2)
    values = [v+np.random.normal(0,s,(v.shape[0],1))
              for v,s in zip(values,noise_std)]

    bases = [HelmholtzBasis(sols,active_speakers[ii])
                   for ii in range(nsources)]
        
    return Benchmark({'samples':samples,'obs':values,
                      'bases':bases,'noise_std':noise_std})

def plot_mfnets_helmholtz_benchmark_data(benchmark):
    from pyapprox.fenics_models.fenics_utilities import \
        get_vertices_of_polygon
    import matplotlib.pyplot as plt
    ampothem,nedges=1.5,8
    vertices = get_vertices_of_polygon(ampothem,nedges)
    vertices = np.hstack([vertices,vertices[:,:1]])
    active_speakers = [[0,1,2,3,4,5,6,7],[2,4,6],[1,3,5,7]]

    fig,axs=plt.subplots(1,3,figsize=(3*8,6))
    for ii in range(3):
        axs[ii].add_artist(plt.Circle((0,0),0.5,fill=False,lw=3))
        axs[ii].add_artist(plt.Circle((-0.5/np.sqrt(8),-0.5/np.sqrt(8)),0.25,fill=False,lw=3))

        for jj in range(len(active_speakers[ii])):
            I = [active_speakers[ii][jj],active_speakers[ii][jj]+1]
            axs[ii].plot(vertices[0,I],vertices[1,I],'k-',lw=5)
        p=axs[ii].scatter(benchmark.samples[ii][0,:],benchmark.samples[ii][1,:],c=np.absolute(benchmark.obs[ii][:,0]),cmap="viridis_r")
        plt.colorbar(p,ax=axs[ii])
        axs[ii].axis('off')
