# Location of bucket on s3
CC_INTERFACE = "" ## Need to set this yourself

# all subjects in pycortex store
SUBJECTS = ["S{}".format(idx) for idx in range(1,12)]

# and their respective transforms
XFMS = dict()
for s in SUBJECTS:
    XFMS[s] = "{}_listening_for_VL".format(s)
