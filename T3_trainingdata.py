import os

biocard_tr=os.listdir("/cis/home/yxie91/nnUNet/nnUNet_raw/Dataset006_axis0/labelsTr")
adni_tr=os.listdir("/cis/home/yxie91/nnUNet/nnUNet_raw/Dataset103_ADNIall/labelsTr")

print(len(biocard_tr))
print(len(adni_tr))

biocard_sub=list(set([f.split('_')[0] for f in biocard_tr]))
adni_sub=list(set([f.split('_')[0] for f in adni_tr]))
print(biocard_sub)
print(adni_sub)

print(len(biocard_sub))
print(len(adni_sub))

print(len(biocard_tr)/len(biocard_sub))
print(len(adni_tr)/len(adni_sub))