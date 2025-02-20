
def LoadFromDisc(filename, amount):
    file = open("DownloadedCrystalProperties/"+filename+".txt","r")
    materials = []
    for x in range(amount): # its from to 0 to "> amount"
        currectword = ""
        while(True):
            cl =file.read(1)
            if cl == ",":
                file.read(1)#skips space after ,
                break
            else:
                currectword += cl

        materials[x].material_id = currectword #only string rest is floats, do some covertation
        currectword = ""
        for y in range(31):
            currectword = ""
        while(True):
            cl =file.read(1)
            if cl == ",":
                file.read(1)#skips space after ,
                break
            else:
                currectword += cl
        
    file.close
    return materials