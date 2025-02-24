#ew i am writing a test volenteerenly
#i can't wait for alex to give this to a proper test frame work with succede and fail if
from APIDownloader import *
from CFILoader import *
#litterly downloads it
ApiDownloader(3,"3test") #because of induction if we can do k+1 we can do unlimted
#propperly needs to do /../ before 3test if we put this in a test folder
testarray = LoadFromDisc("3test",3)
#if these prints "mp-xxxxxx" and x being random numbers test is complete
print(testarray[0].material_id + "\n")
print(testarray[1].material_id + "\n")
print(testarray[2].material_id + "\n")

#as of 10:37 24/02 don't run this i man missing 2-4 lines of code cuz shit
#nvm i just made it look stupid when read from human form but eaiser to code

#i should write delete "3test" file but long live overwrite functions