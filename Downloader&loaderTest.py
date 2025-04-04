#ew i am writing a test volenteerenly
#i can't wait for alex to give this to a proper test frame work with succede and fail if
from data_retrival import *


if False:
    data = load_data_local("Mads100")
    print(data[99][0].material_id)   
else:
    data = get_materials_data(100)
    print(f"Retrieved {len(data)} entries")
    save_data_local("Mads100", data)

    with open("DownloadedCrystalProperties/Mads100.txt") as f:
        print(f"File has {sum(1 for _ in f)} lines")
     
    #litterly downloads it
 #save_data_local("100test",get_materails_data(100)) #because of induction if we can do k+1 we can do unlimted
#propperly needs to do /../ before 3test if we put this in a test folder
 #testarray = load_data_local("100test",100)
#if these prints "mp-xxxxxx" and x being random numbers test is complete
 #print(testarray[0].material_id + str(testarray[0].is_metal) + "\n")
 #print(testarray[1].material_id + str(testarray[1].is_metal) + "\n")
 #print(testarray[2].material_id + testarray[2].is_metal + "\n")

#as of 10:37 24/02 don't run this i man missing 2-4 lines of code cuz shit
#nvm i just made it look stupid when read from human form but eaiser to code

#i should write delete "3test" file but long live overwrite functions