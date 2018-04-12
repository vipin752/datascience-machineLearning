import os
import re as re
    
def readFile(filename):
      filehandle = open(filename,encoding='utf8')
      #print(filehandle.read())
      for line in filehandle:
          line = line.rstrip('\n')
          #print(line)
          #list=line.split()
          #for word in list:
          #print(line)
          x=re.search("^(0[1-9]|1[0-9]| 2[0-9]|3[0-1])(.|-,/)(0[1-9]|1[0-2])(.|-,/)([1-9][0-9][0-9][0-9])$",line.strip())
          x1=re.search("^([1-9][0-9][0-9][0-9])(.|-,/)(0[1-9]|1[0-2])(.|-,/)(0[1-9]|1[0-9]| 2[0-9]|3[0-1])$",line.strip())
          x2=re.search("^([1-9][0-9][0-9][0-9])(.|-,/)(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)(.|-,/)(0[1-9]|1[0-9]| 2[0-9]|3[0-1])$",line.strip())
          x3=re.search("^(0[1-9]|1[0-9]|2[0-9]|3[0-1])(.|-,/)(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)(.|-,/)([1-9][0-9][0-9][0-9])$",line.strip())
          x4=re.search("^([1-9]|1[0-9]|2[0-9]|3[0-1])(.|-,/)(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)(.|-,/)([1-9][0-9][0-9][0-9])$",line.strip())
          x5=re.search("^([1-9]|1[0-9]|2[0-9]|3[0-1])(.|-,/)([1-9]|1[0-2])(.|-,/)([1-9][0-9][0-9][0-9])$",line.strip())
          x6=re.search("^(January|February|March|April|May|June|July|August|September|October|November|December) ([1-9][0-9][0-9][0-9])$",line.strip())
          x7=re.search("^(0[1-9]|1[0-9]|2[0-9]|3[0-1]) (January|February|March|April|May|June|July|August|September|October|November|December) ([1-9][0-9][0-9][0-9])$",line.strip())
          x8=re.search("^([1-9]|1[0-9]|2[0-9]|3[0-1])(.|-,/)(0[1-9]|1[0-2])(.|-,/)([1-9][0-9][0-9][0-9])$",line.strip())
          x9=re.search("^(0[0-9]|1[0-2]|2[0-3])(:)(0[0-9]|1[0-9]|2[0-9]|3[0-9]|4[0-9]|5[0-9])$",line.strip())
          
          if x:
              print(x.group())
          elif x1:
              print(x1.group())
          elif x2:
              print(x2.group())
          elif x3:
              print(x3.group())
          elif x4:
              print(x4.group())
          elif x5:
              print(x5.group())
          elif x6:
              print(x6.group())
          elif x7:
              print(x7.group())
          elif x8:
              print(x8.group())
          elif x9:
              print(x9.group())
          else:
              # Print a separator, without a newline character.
              print(' ', end='')
         # Print the original line, without a newline character.
      #print(line, end='')
    
        # Print the last newline character.
      print()
      filehandle.close()
path=os.getcwd();
path=path+"\Assignment1.txt"
filename = os.path.join(path)
print(filename)
readFile(filename)