"""
1. Tell python where to find the appropriate functions.
"""

from copy import deepcopy
import nipype.interfaces.io as nio           # Data i/o 
import nipype.interfaces.spm as spm          # spm
import nipype.interfaces.fsl as fsl          # fsl
import nipype.interfaces.freesurfer as fs    # freesurfer
import nipype.pipeline.engine as pe          # pypeline engine
import os                                    # system functions
import nipype.interfaces.utility as util     # misc. modules
import nipype.algorithms.rapidart as ra	     # ra
import nipype.interfaces.matlab as mlab
import numpy as np

#####################################################################
"""
2. Setup any package specific configuration. The output file format
   for FSL routines is being set to uncompressed NIFTI and a specific
   version of matlab is being used. The uncompressed format is
   required because SPM does not handle compressed NIFTI.
"""

# Tell freesurfer what subjects directory to use
#basedir = '/mindhive/gablab/rtsmoking/'
base_dir = '/speechlab/2/jsegawa/SEQ03/'
basedir = '/speechlab/2/jsegawa/SEQ03/'

#data_dir = os.path.join(basedir+'data')
data_dir = os.path.join(base_dir+'niftidata')

output_dir = os.path.join(base_dir,'nipype')

#subjects_dir = os.path.join(basedir+'surfaces')
subjects_dir = os.path.join(base_dir+'subjects_nipype')

fs.FSCommand.set_default_subjects_dir(subjects_dir)
fsl.FSLCommand.set_default_output_type('NIFTI_GZ')

# Set the way matlab should be called
mlab.MatlabCommand.set_default_matlab_cmd("matlab -nodesktop -nosplash")
# If SPM is not in your MATLAB path you should add it here
mlab.MatlabCommand.set_default_paths('/speechlab/software/spm8')

from fsl_flow import l1pipeline

preproc = l1pipeline.get_node('preproc')
## preproc.disconnect(preproc.get_node('extractref'),'roi_file',
##                   preproc.get_node('realign'),'ref_file')

"""
3. The following lines of code sets up the necessary information
   required by the datasource module. It provides a mapping between
   run numbers (nifti files) and the mnemonic ('struct', 'func',
   etc.,.)  that particular run should be called. These mnemonics or
   fields become the output fields of the datasource module. In the
   example below, run 'f3' is of type 'func'. The 'f3' gets mapped to
   a nifti filename through a template '%s.nii'. So 'f3' would become
   'f3.nii'.
"""

#subject_list = ['s1_1_SST', 's1_2_MID',
#                's2_2_MID',
#                's3_1_MID', 's3_2_SST',
#                's4_1_MID']

#subjectinfo = dict(s2_1_SST=['N','T','T','T','N'],
#                   )

subject_list = ['SEQ03001', 'SEQ03002', 'SEQ03003',
                'SEQ03005', 'SEQ03006', 'SEQ03009',
                'SEQ03010', 'SEQ03011', 'SEQ03012',
                'SEQ03013', 'SEQ03015', 'SEQ03018']




infosource = pe.Node(interface=util.IdentityInterface(fields=['subject_id']), name="infosource")

"""Here we set up iteration over all the subjects. The following line
is a particular example of the flexibility of the system.  The
``datasource`` attribute ``iterables`` tells the pipeline engine that
it should repeat the analysis on each of the items in the
``subject_list``. In the current example, the entire first level
preprocessing and estimation will be repeated for each subject
contained in subject_list.
"""

infosource.iterables = ('subject_id', subject_list)

######################################################################
# Setup preprocessing pipeline nodes

"""
Now we create a :class:`nipype.interfaces.io.DataGrabber` object and
fill in the information from above about the layout of our data.  The
:class:`nipype.pipeline.NodeWrapper` module wraps the interface object
and provides additional housekeeping and pipeline specific
functionality.
"""

datasource = pe.Node(interface=nio.DataGrabber(infields=['subject_id'],#,'sesstype'],
                                               outfields=['func', 'struct']),
                                                         # 'ref','roi']),
                     name = 'datasource')
datasource.inputs.base_directory = data_dir
datasource.inputs.template = '*'
datasource.inputs.field_template = dict(func='%s/functionals/%s*.nii.gz',
                                        struct='%s/mprage/%s*.nii.gz')
     #ref='/data/reward/smoking/rt_gro_data/rtSmoking/*/%s/xfm/study_ref.nii',
     #roi='/data/reward/smoking/rt_gro_data/rtSmoking/*/%s/mask/%s.nii')
# Map field names to individual subject runs.
info = dict(func=[['subject_id', 'subject_id']],
            struct=[['subject_id','subject_id']])#,
            #ref=[['subject_id']],
            #roi=[['subject_id']])  #roi=[['subject_id','sesstype']])
datasource.inputs.template_args = info

"""
def gettype(sid):
    if sid.endswith('MID'):
        return 'mid'
    else:
        return 'ss'
"""

#######################################################################
# setup analysis components
#######################################################################
"""
   a. Setup a function that returns subject-specific information about
   the experimental paradigm. This is used by the
   :class:`nipype.interfaces.model.SpecifyModel` to create the
   information necessary to generate an SPM design matrix. In this
   tutorial, the same paradigm was used for every participant. Other
   examples of this function are available in the `doc/examples`
   folder. Note: Python knowledge required here.
"""
from nipype.interfaces.base import Bunch
from copy import deepcopy
import numpy as np
import scipy.io as sio

funcinfo = {}
funcinfo['SEQ03001'] = ['00','01','02','03','04','05','06']
funcinfo['SEQ03002'] = ['00','01','02','03','04','05','06']
funcinfo['SEQ03003'] = ['00','01','02','03','04','05','06']
funcinfo['SEQ03005'] = ['00','01','02','03','04','05']
funcinfo['SEQ03006'] = ['00','01','02','03','04','05']
funcinfo['SEQ03009'] = ['00','01','02','03','04','05','06','07']
funcinfo['SEQ03010'] = ['00','01','02','03','04','05','06','07']
funcinfo['SEQ03011'] = ['00','01','02','03','04','05','06']
funcinfo['SEQ03012'] = ['00','01','02','03','04']
funcinfo['SEQ03013'] = ['00','01','02','03','04','05','06','07']
funcinfo['SEQ03015'] = ['00','01','02','03','04','05','06','07']
funcinfo['SEQ03018'] = ['00','01','02','03','04','05','06','07']

all_subj = funcinfo.keys()
all_subj.sort()

"""
def getntrials(subject_id,runno):
    s_idx = all_subj.index(subject_id)
    runinfo_file = '/speechlab/software/SLT/scripts/jsegawa/runinfo_SEQ03_nipype_regressbad.mat'
    runinfo_data = sio.loadmat(runinfo_file)
    ntrials = 0
    for c in runinfo_data['onsets_secs'][0][s_idx][0][runno][0].tolist():
        ntrials += len(c[0].tolist())
    return ntrials
"""

# build subject-specific information
def subjectinfo(subject_id):
    print "Subject ID: %s" % str(subject_id)
    s_idx = all_subj.index(subject_id)
    print "Subject index: %d" % s_idx
    nruns = len(funcinfo[subject_id])
    print "No. of runs: %d\n" % nruns
    output = []
    runinfo_file = '/speechlab/software/SLT/scripts/jsegawa/runinfo_SEQ03_nipype_regressbad.mat'
    runinfo_data = sio.loadmat(runinfo_file)
    for r in range(nruns):
        onset = []
        condnames = ['Baseline','Ill,Learned','Legal','Ill,Novel']
        #print "r: %d" % r
        """
        regressor_names=None
        regressors=None
        for o in runinfo_data['badvols_secs'][0][s_idx][0][r][0].tolist():
            if not regressor_names:
                regressor_names = []
            if not regressors:
                regressors = []
            regressor_names.append('badvol%d'%o)
            badvollist = np.zeros(41)  #getntrials(subject_id,r))
            badvollist[o-1] = 1   #badvollist[o-1] = 1
            regressors.append(badvollist.tolist())
        """
        for c in range(len(condnames)):
            onset.append(runinfo_data['onsets_secs'][0][s_idx][0][r][0][c][0].tolist())
            #print "1st scan: %d" % onset[0][0]
        output.insert(r,
                      Bunch(conditions=condnames,
                            onsets=deepcopy(onset),
                            durations=[[0] for s in condnames], #[[2.5] for s in condnames],
                            amplitudes=None,
                            tmod=None,
                            pmod=None,
                            regressor_names=None, #regressor_names,
                            regressors=None)) #regressors))
    return output

"""
   b. Setup the contrast structure that needs to be evaluated. This is
   a list of lists. The inner list specifies the contrasts and has the
   following format - [Name,Stat,[list of condition names],[weights on
   those conditions]. The condition names must match the `names`
   listed in the `subjectinfo` function described above. 
"""
cont1 = ['All-baseline','T', ['Baseline','Ill,Learned','Legal','Ill,Novel'],[-1,0.333,0.333,0.333]]
cont2 = ['Illegal,novel-learned','T', ['Ill,Learned','Ill,Novel'],[-1,1]]
contf1 = ['regf','F', [cont1, cont2]]
contrasts = [cont1,cont2,contf1]


"""
Set preprocessing parameters
----------------------------
"""

smoothvalnode = l1pipeline.get_node('preproc.smoothval')
assert(str(smoothvalnode)=='preproc.smoothval')
smoothvalnode.iterables = ('fwhm', [0,6])

art = l1pipeline.inputs.preproc.art
art.use_differences      = [True,False]
#art.use_differences      = [True,True] #this is what carrie's script has
#first is use scan to scan for motion, second is use overall mean for intensity
art.use_norm             = True
#composite measure of motion
art.norm_threshold       = 1
#in mm
art.zintensity_threshold = 3
#in standard dev
art.parameter_source = 'FSL'

"""
Set up node specific inputs
---------------------------

We replicate the modelspec parameters separately for the surface- and
volume-based analysis.
"""

TR = 10
hpcutoff = 400

l1pipeline.inputs.preproc.highpass.op_string = '-bptf %.10f -1'%(hpcutoff/(2*TR))

l1pipeline.inputs.preproc.fssource.subjects_dir = subjects_dir
l1pipeline.inputs.modelfit.modelspec.input_units = 'secs' #'scans'
l1pipeline.inputs.modelfit.modelspec.output_units = 'secs' #'scans'
l1pipeline.inputs.modelfit.modelspec.time_repetition = TR
l1pipeline.inputs.modelfit.modelspec.high_pass_filter_cutoff = hpcutoff #np.inf

#adding these for sparse acquisition
l1pipeline.inputs.modelfit.modelspec.time_acquisition = 2.5
l1pipeline.inputs.modelfit.modelspec.model_hrf = True
l1pipeline.inputs.modelfit.modelspec.scan_onset = 4 #start of scan relative to onset of run (sec)
l1pipeline.inputs.modelfit.modelspec.volumes_in_cluster = 1
l1pipeline.inputs.modelfit.modelspec.stimuli_as_impulses = True
l1pipeline.inputs.modelfit.modelspec.is_sparse = True


l1pipeline.inputs.modelfit.level1design.interscan_interval = TR
l1pipeline.inputs.modelfit.level1design.bases = {'hrf':{'derivs':False}} #{'dgamma':{'derivs': True}} 

l1pipeline.inputs.overlay.overlaystats.stat_thresh = (3, 7)

l1pipeline.inputs.inputnode.contrasts = contrasts
l1pipeline.inputs.inputnode.surf_dir = subjects_dir


l1pipeline.get_node('modelfit.conestimate').iterfield.append('fcon_file')
l1pipeline.get_node('modelfit').connect(l1pipeline.get_node('modelfit.modelgen'),
                                        'fcon_file',
                                        l1pipeline.get_node('modelfit.conestimate'),
                                        'fcon_file')

#################################################################################
# Setup pipeline
#################################################################################

"""
   The nodes setup above do not describe the flow of data. They merely
   describe the parameters used for each function. In this section we
   setup the connections between the nodes such that appropriate
   outputs from nodes are piped into appropriate inputs of other
   nodes.  

   a. Use :class:`nipype.pipeline.engine.Pipeline` to create a
   graph-based execution pipeline for first level analysis. The config
   options tells the pipeline engine to use `workdir` as the disk
   location to use when running the processes and keeping their
   outputs. The `use_parameterized_dirs` tells the engine to create
   sub-directories under `workdir` corresponding to the iterables in
   the pipeline. Thus for this pipeline there will be subject specific
   sub-directories. 

   The ``nipype.pipeline.engine.Pipeline.connect`` function creates the
   links between the processes, i.e., how data should flow in and out
   of the processing nodes. 
"""
"""
Setup the pipeline
------------------

The nodes created above do not describe the flow of data. They merely
describe the parameters used for each function. In this section we
setup the connections between the nodes such that appropriate outputs
from nodes are piped into appropriate inputs of other nodes.

Use the :class:`nipype.pipeline.engine.Workfow` to create a
graph-based execution pipeline for first level analysis. 
"""

level1 = pe.Workflow(name="level1")
level1.base_dir = os.path.join(basedir,'workingdir/realtime')
level1.config['crashdumps_dir'] = os.path.join(basedir,'workingdir/crashes')
"""
level1.connect([(infosource, datasource, [('subject_id', 'subject_id'),
                                          (('subject_id', gettype), 'sesstype')]),
                (datasource,l1pipeline,[('func','inputnode.func'),
                                        ('ref','preproc.realign.ref_file')]),
                (infosource,l1pipeline,[('subject_id','inputnode.subject_id'),
                                        ('subject_id','inputnode.fssubject_id'),
                                        (('subject_id', subjectinfo),
                                         'inputnode.session_info'),
                                        ]),
                ])
"""
level1.connect([(infosource, datasource, [('subject_id', 'subject_id')]),
                                          ##(('subject_id', gettype), 'sesstype')]),
                (datasource,l1pipeline,[('func','inputnode.func')]),
                                        ##('ref','preproc.realign.ref_file')]),
                (infosource,l1pipeline,[('subject_id','inputnode.subject_id'),
                                        ('subject_id','inputnode.fssubject_id'),
                                        (('subject_id', subjectinfo),
                                         'inputnode.session_info'),
                                        ]),
                ])

def getsubs(subject_id):
    subs = [('_subject_id_%s/'%subject_id,''),
            ('_plot_type_','')]
    for i, con in enumerate(contrasts):
        subs.append(('_ztop%d'%i, con[0]))
        subs.append(('_slicestats%d/tstat1_overlay'%i, con[0]))
        subs.append(('_plot_motion%d'%i, ''))
    return subs


"""
Setup the datasink
"""
datasink = pe.Node(interface=nio.DataSink(container='social'), name="datasink")
datasink.inputs.base_directory = os.path.join(basedir,'l1output/realtime')

# store relevant outputs from various stages of the 1st level analysis
level1.connect([(infosource, datasink,[('subject_id','container'),
                                       (('subject_id', getsubs), 'substitutions'),
                                       ]),
                #(datasource, datasink,[('roi','mask'),
                #                       ]),
                (l1pipeline, datasink,[('modelfit.conestimate.copes','stats.@copes'),
                                       ('modelfit.conestimate.varcopes','stats.@varcopes'),
                                       ('modelfit.conestimate.tstats','stats.@tstats'),
                                       ('modelfit.conestimate.zstats','stats.@zstats'),
                                       ('modelfit.ztop.out_file','stats.@pvals'),
                                       ('preproc.meanfunc2.out_file','meanfunc'),
                                       ('preproc.realign.par_file','motion'),
                                       ('preproc.plot_motion.out_file','motion.@plots'),
                                       ('preproc.art.statistic_files','art.@stats'),
                                       ('preproc.art.outlier_files','art.@outliers'),
                                       ])
                ])
#                                       ('overlay.slicestats.out_file','overlay'),
#                                       ('preproc.surfregister.out_reg_file','bbreg'),

##########################################################################
# Execute the pipeline
##########################################################################

"""
   The code discussed above sets up all the necessary data structures
   with appropriate parameters and the connectivity between the
   processes, but does not generate any output. To actually run the
   analysis on the data the ``nipype.pipeline.engine.Pipeline.Run``
   function needs to be called. 
"""
if __name__ == '__main__':
    level1.run()
    level1.write_graph(graph2use='flat')
