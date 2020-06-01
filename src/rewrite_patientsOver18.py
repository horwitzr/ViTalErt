# Reads each of csvfiles and keeps rows with patientUnitStayIdOver18 for
# patients over 18. Writes trimmed data to dir_data_over18
def keepPtsOver18(csvfiles, patientUnitStayIdOver18, dir_data_over18, dir_data_raw):
  import pandas as pd
  import os
  csvfiles.remove(dir_data_raw + 'hospital.csv') # no patientunitstayid
  for csvfile in csvfiles:
      print('------')
      print('Working on ' + csvfile)
      data_raw = pd.read_csv(csvfile)
      data_over18 = data_raw[data_raw['patientunitstayid'].isin(patientUnitStayIdOver18)]
      base = os.path.basename(csvfile)
      print(base)
      newFileNameWithPath = dir_data_over18 + base
      data_over18.to_csv(newFileNameWithPath, index=False)
      print('Saved data to ' + newFileNameWithPath)
      print('------')
