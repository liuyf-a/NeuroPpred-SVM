sed -n '1, 388p' ./mn_CLS.txt > pos_fold1.txt
sed -n '389, 776p' ./mn_CLS.txt > pos_fold2.txt
sed -n '777, 1164p' ./mn_CLS.txt > pos_fold3.txt
sed -n '1165, 1552p' ./mn_CLS.txt > pos_fold4.txt
sed -n '1553, 1940p' ./mn_CLS.txt > pos_fold5.txt
sed -n '1941, 2328p' ./mn_CLS.txt > neg_fold1.txt
sed -n '2329, 2716p' ./mn_CLS.txt > neg_fold2.txt
sed -n '2717, 3104p' ./mn_CLS.txt > neg_fold3.txt
sed -n '3105, 3492p' ./mn_CLS.txt > neg_fold4.txt
sed -n '3493, 3880p' ./mn_CLS.txt > neg_fold5.txt
cat pos_fold1.txt neg_fold1.txt > ./fold1/test.txt 
cat pos_fold2.txt pos_fold3.txt pos_fold4.txt pos_fold5.txt neg_fold2.txt neg_fold3.txt neg_fold4.txt neg_fold5.txt > ./fold1/train.txt 
cat pos_fold2.txt neg_fold2.txt > ./fold2/test.txt
cat pos_fold1.txt pos_fold3.txt pos_fold4.txt pos_fold5.txt neg_fold1.txt neg_fold3.txt neg_fold4.txt neg_fold5.txt > ./fold2/train.txt
cat pos_fold3.txt neg_fold3.txt > ./fold3/test.txt
cat pos_fold1.txt pos_fold2.txt pos_fold4.txt pos_fold5.txt neg_fold1.txt neg_fold2.txt neg_fold4.txt neg_fold5.txt > ./fold3/train.txt
cat pos_fold4.txt neg_fold4.txt > ./fold4/test.txt
cat pos_fold1.txt pos_fold2.txt pos_fold3.txt pos_fold5.txt neg_fold1.txt neg_fold2.txt neg_fold3.txt neg_fold5.txt > ./fold4/train.txt
cat pos_fold5.txt neg_fold5.txt > ./fold5/test.txt
cat pos_fold1.txt pos_fold2.txt pos_fold3.txt pos_fold4.txt neg_fold1.txt neg_fold2.txt neg_fold3.txt neg_fold4.txt > ./fold5/train.txt
