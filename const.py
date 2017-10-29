

SAMPLE_FREQ = 50
FILE_MARGINES = 5*SAMPLE_FREQ  ## number of samples to ignore in the  start and in the end of the file (5 seconds )
WINDOW_SIZE = 128  ## sliding window size
PEAKS_WINDOW_SIZE = 1 * WINDOW_SIZE  ## sliding window size for peaks count feature

DEVICE_MODE_LABELS = ['pocket','swing','texting','talking','whatever']
USER_MODE_LABELS = ['walking','fastwalking','stairs','static','whatever']


## Features by categories
##----------------------------------------------------------------------------------------------------------------------

## basic g-force  features :
ACC_FEATURES = ['agforce',  ## avarage
                'mgforce',
                'vgforce',  ## variance
                'iqrforce',  ## iqr
                'entforce',  ## entropy
                'skforce',  ## skewness
                'kuforce',  ## kurtosis
                'maxgforce',
                'mingforce',  ## min
                'mingforceabs',  ## abs min
                'aadforce',  ## average absolute dfference
                'smaforce',  ## signal magnitude area
                'gcrgforce',  ## g-crossing rate
                'mcrgforce', ## medain-crossing rate
                'peaksgforce',  ## peaks count in PEAKS_WINDOW_SIZE
                'enyforce',  ## signal energy
                'ampgforce']

## basic gyro features :
GYRO_FEATURES = ['agyro' ,  ## avarage
                 'mgyro',  ## median
                 'vgyro',  ## variance
                 'iqqgyro',  ## iqr
                 'entgyro',  ## entropy
                 'skgyro',  ## skewness
                 'kugyro',  ## kurtosis
                 'maxgyro',  ## max
                 'maxgyroabs',     ## abs max
                 'mingyro',  ## min
                 'aadgyro',  ## average absolute dfference
                 'smagyro',  ## signal magnitude area
                 'mcrgyro',  ## medain-crossing rate
                 'peaksgyro',  ## peaks count in PEAKS_WINDOW_SIZE
                 'enygyro',  ## signal energy
                 'amAccGyro',  ## amplitude Acc Gyro
                 'ampgyro']  ## amplitude |max - min|

## components based gyro features :
COMPONENTS_FEATURES = ['afxforce', 'afyforce', 'afzforce',  ## avarage
                       'mfxforce', 'mfyforce', 'mfzforce',  ## median
                       'vfxforce', 'vfyforce', 'vfzforce',  ## variance
                       'maxfxforce', 'maxfyforce', 'maxfzforce',  ## max
                       'minfxforce', 'minfyforce', 'minfzforce',  ## min
                       'aadfxforce', 'aadfyforce', 'aadfzforce',  ## average absolute dfference
                       'smafxforce', 'smafyforce', 'smafzforce',  ## signal magnitude area
                       'gcrfxforce', 'gcrfyforce', 'gcrfzforce',  ## g-crossing rate
                       'mcrfxforce', 'mcrfyforce', 'mcrfzforce',  ## medain-crossing rate
                       'peakfxforce', 'peakfyforce', 'peakfzforce',
                       'enyfxforce', 'enyfyforce', 'enyfzforce',  ## signal energy
                       'ampfxforce', 'ampfyforce', 'ampfzforce',
                       'ampwxgyro', 'ampwygyro', 'ampwzgyro']

## cross sensors features :
CROSS_SENSORS_FEATURES = ['gforcegyrocorr',  ## correlation
                          'MultiGyroAcc',  ## max  gyro * max acc
                          'MultiVarGyroAcc', ## variance  gyro * variance acc
                          'ampGyrofxforce', 'ampGyrofyforce', 'ampGyrofzforce',  ## amplitude Acc Gyro
                          'ampGyrowxforce', 'ampGyrowyforce', 'ampGyrowzforce'  ## amplitude Acc Gyro
                         ]

LIGHT_FEATURES = ['alight',  ## avarage
                  'mlight',  ## median
                  'vlight',  ## variance
                  'iqrlight',  ## iqr
                  'MultiGyroLight', 'MultiLightAcc',  ## max  gyro * max acc
                  'MultiVarGyroLight', 'MultiVarLightAcc',  ## variance  gyro * variance acc
                  'maxlight',  ## max
                  'minlight',  ## min
                  'aadlight',  ## average absolute dfference
                  'smalight',  ## signal magnitude area
                  'mcrlight',  ## medain-crossing rate
                  'peakslight',  ## peaks count in PEAKS_WINDOW_SIZE
                  'enylight',  ## signal energy
                  'amAccLight', 'amLightGyro',  ## amplitude Acc Gyro
                  'amplight']  ## amplitude |max - min|

BAROMETER_FEATURES = ['ampbaro', 'vbaro', 'mcrbaro', 'enybaro', 'smabaro',  ## baro features
                      'amAccBaro', 'amBaroGyro', 'MultiVarBaroAcc', 'MultiVarBaroGyro']  ## baro cross sensor features

FEATURES = ACC_FEATURES + GYRO_FEATURES + COMPONENTS_FEATURES + CROSS_SENSORS_FEATURES