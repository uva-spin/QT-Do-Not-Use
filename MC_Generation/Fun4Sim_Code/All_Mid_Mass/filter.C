# include "TFile.h"
# include "TTree.h"
# include "TH1F.h"
# include "TCanvas.h"

using namespace std;

void filter()
{

//variable
float D0U_drift[100];
  float D0Up_drift[100];
  float D0X_drift[100];
  float D0Xp_drift[100];
  float D0V_drift[100];
  float D0Vp_drift[100];

  float D2U_drift[100];
  float D2Up_drift[100];
  float D2X_drift[100];
  float D2Xp_drift[100];
  float D2V_drift[100];
  float D2Vp_drift[100];

  float D3pU_drift[100];
  float D3pUp_drift[100];
  float D3pX_drift[100];
  float D3pXp_drift[100];
  float D3pV_drift[100];
  float D3pVp_drift[100];

  float D3mU_drift[100];
  float D3mUp_drift[100];
  float D3mX_drift[100];
  float D3mXp_drift[100];
  float D3mV_drift[100];
  float D3mVp_drift[100];

  float D0U_ele[100];
  float D0Up_ele[100];
  float D0X_ele[100];
  float D0Xp_ele[100];
  float D0V_ele[100];
  float D0Vp_ele[100];

  float D2U_ele[100];
  float D2Up_ele[100];
  float D2X_ele[100];
  float D2Xp_ele[100];
  float D2V_ele[100];
  float D2Vp_ele[100];

  float D3pU_ele[100];
  float D3pUp_ele[100];
  float D3pX_ele[100];
  float D3pXp_ele[100];
  float D3pV_ele[100];
  float D3pVp_ele[100];

  float D3mU_ele[100];
  float D3mUp_ele[100];
  float D3mX_ele[100];
  float D3mXp_ele[100];
  float D3mV_ele[100];
  float D3mVp_ele[100];

 int P1Y1_ele[500];
  int P1Y2_ele[500];
  int P1X1_ele[500];
  int P1X2_ele[500];
  int P2X1_ele[500];
  int P2X2_ele[500];
  int P2Y1_ele[500];
  int P2Y2_ele[500];



int H1B_ele[100];
int n_tracks[100];

float limit = 1000.00;

//source file
TChain *ttree;
ttree = new TChain("QA_ana");
char name_file[500];

ttree->Add("trackQA_v2.root");


TTree *T = (TTree*)gROOT->FindObject("QA_ana");
//
T->SetBranchAddress("D0U_drift",    &D0U_drift);
  T->SetBranchAddress("D0Up_drift",    &D0Up_drift);
  T->SetBranchAddress("D0X_drift",    &D0X_drift);
  T->SetBranchAddress("D0Xp_drift",    &D0Xp_drift);
  T->SetBranchAddress("D0V_drift",    &D0V_drift);
  T->SetBranchAddress("D0Vp_drift",    &D0Vp_drift);
  
  T->SetBranchAddress("D2U_drift",    &D2U_drift);
  T->SetBranchAddress("D2Up_drift",    &D2Up_drift);
  T->SetBranchAddress("D2X_drift",    &D2X_drift);
  T->SetBranchAddress("D2Xp_drift",    &D2Xp_drift);
  T->SetBranchAddress("D2V_drift",    &D2V_drift);
  T->SetBranchAddress("D2Vp_drift",    &D2Vp_drift);
  
  T->SetBranchAddress("D3pU_drift",    &D3pU_drift);
  T->SetBranchAddress("D3pUp_drift",    &D3pUp_drift);
  T->SetBranchAddress("D3pX_drift",    &D3pX_drift);
  T->SetBranchAddress("D3pXp_drift",    &D3pXp_drift);
  T->SetBranchAddress("D3pV_drift",    &D3pV_drift);
  T->SetBranchAddress("D3pVp_drift",    &D3pVp_drift);
  
  T->SetBranchAddress("D3mU_drift",    &D3mU_drift);
  T->SetBranchAddress("D3mUp_drift",    &D3mUp_drift);
  T->SetBranchAddress("D3mX_drift",    &D3mX_drift);
  T->SetBranchAddress("D3mXp_drift",    &D3mXp_drift);
  T->SetBranchAddress("D3mV_drift",    &D3mV_drift);
  T->SetBranchAddress("D3mVp_drift",    &D3mVp_drift);

  T->SetBranchAddress("D0U_ele",    &D0U_ele);
  T->SetBranchAddress("D0Up_ele",    &D0Up_ele);
  T->SetBranchAddress("D0X_ele",    &D0X_ele);
  T->SetBranchAddress("D0Xp_ele",    &D0Xp_ele);
  T->SetBranchAddress("D0V_ele",    &D0V_ele);
  T->SetBranchAddress("D0Vp_ele",    &D0Vp_ele);
  
  T->SetBranchAddress("D2U_ele",    &D2U_ele);
  T->SetBranchAddress("D2Up_ele",    &D2Up_ele);
  T->SetBranchAddress("D2X_ele",    &D2X_ele);
  T->SetBranchAddress("D2Xp_ele",    &D2Xp_ele);
  T->SetBranchAddress("D2V_ele",    &D2V_ele);
  T->SetBranchAddress("D2Vp_ele",    &D2Vp_ele);
  
  T->SetBranchAddress("D3pU_ele",    &D3pU_ele);
  T->SetBranchAddress("D3pUp_ele",    &D3pUp_ele);
  T->SetBranchAddress("D3pX_ele",    &D3pX_ele);
  T->SetBranchAddress("D3pXp_ele",    &D3pXp_ele);
  T->SetBranchAddress("D3pV_ele",    &D3pV_ele);
  T->SetBranchAddress("D3pVp_ele",    &D3pVp_ele);
  
  T->SetBranchAddress("D3mU_ele",    &D3mU_ele);
  T->SetBranchAddress("D3mUp_ele",    &D3mUp_ele);
  T->SetBranchAddress("D3mX_ele",    &D3mX_ele);
  T->SetBranchAddress("D3mXp_ele",    &D3mXp_ele);
  T->SetBranchAddress("D3mV_ele",    &D3mV_ele);
  T->SetBranchAddress("D3mVp_ele",    &D3mVp_ele);

T->SetBranchAddress("P1Y1_ele",    &P1Y1_ele);
T->SetBranchAddress("P1Y2_ele",    &P1Y2_ele);
T->SetBranchAddress("P1X1_ele",    &P1X1_ele);
T->SetBranchAddress("P1X2_ele",    &P1X2_ele);
T->SetBranchAddress("P2X1_ele",    &P2X1_ele);
T->SetBranchAddress("P2X2_ele",    &P2X2_ele);
T->SetBranchAddress("P2Y1_ele",    &P2Y1_ele);
T->SetBranchAddress("P2Y2_ele",    &P2Y2_ele);

TFile *newfile = new TFile( "rawhit_clean.root","RECREATE");

TTree *newfiletree = T->CloneTree(0);



for (Long64_t ievt=0; ievt< T->GetEntries();ievt++) 
{
      T->GetEntry(ievt);
      if (D0X_drift[0] < 1000. && D0X_drift[1] < 1000. && D2X_drift[0] < 1000. && D2X_drift[1] < 1000. && (D3pX_drift[0] <limit || D3mX_drift[0] < limit) && (D3pX_drift[1] <1000. || D3mX_drift[1] < limit) && (P1Y1_ele[0]<1000 || P1Y2_ele[0]<1000 || P1X1_ele[0]<1000 || P1X2_ele[0]<1000 || P2X1_ele[0]<1000 || P2X2_ele[0]<1000 || P2Y1_ele[0]<1000 || P2Y2_ele[0]<1000) && (P1Y1_ele[1]<1000 || P1Y2_ele[1]<1000 || P1X1_ele[1]<1000 || P1X2_ele[1]<1000 || P2X1_ele[1]<1000 || P2X2_ele[1]<1000 || P2Y1_ele[1]<1000 || P2Y2_ele[1]<1000)) 
     
       {
        newfiletree->Fill();
       }
}

newfiletree->Write();



}
