/// Macro to display the detector geometry.
/**
 * "geom.root" should be created beforehand, typically by running "Fun4Sim.C".
 * Then you can just execute "root disp_geom.C".
 */
void disp_geom()
{
  gGeoManager = TGeoManager::Import("geom.root");
  //gGeoManager = TGeoManager::Import("GenFitExtrapolatorGeom.root");
  //gGeoManager->Export("geom.root");
  TGeoNode *current = gGeoManager->GetCurrentNode();
  current->GetVolume()->Draw("ogl");
}
