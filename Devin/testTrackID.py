import ROOT
 
def decode_source_flag(encoded):
    return (encoded >> 5) & 0x3  # Extract bits 5-6
 
def decode_process_id(encoded):
    return encoded & 0x1F  # Extract lower 5 bits
 
file = ROOT.TFile("JPsi_Dump.root") 
tree = file.Get("tree")
 
for event in tree:
    print("EventID:", event.eventID)
    for i in range(len(event.gpz)):
        for j in range(len(event.hitTrackID)):
            #check for the track id. When the track id matches with the track id at the hit level, we access the hit information.
            #print("hitTrackID: ", event.hitTrackID)
            if event.hitTrackID[j] != event.gTrackID[i]:
                continue
            hit_id = event.hitID[j]
            hit_trackid = event.hitTrackID[j]
            encoded_value = event.gProcessID[j] 
            process_id = decode_process_id(encoded_value) #decoding the process_id
            source_flag = decode_source_flag(encoded_value) #decoding the source_flag

            print("hitID:", hit_id)
            print("hit_trackID:", hit_trackid)
            print("Encoded value:", encoded_value)
            print("Decoded Process ID:", process_id)
            print("Decoded Source Flag:", source_flag)
        print("-----Next Track------------")
    print("========Next Event=============")
