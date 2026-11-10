with Ada.Command_Line;
with Test_Activations;
with Test_Framework;
with Test_Gradient_Checking;
with Test_Layers;
with Test_Losses;
with Test_Matrices;
with Test_Neuron;
with Test_Random;
with Test_Vectors;

procedure Test_Runner is
begin
   Test_Vectors.Run;
   Test_Matrices.Run;
   Test_Activations.Run;
   Test_Losses.Run;
   Test_Random.Run;
   Test_Neuron.Run;
   Test_Layers.Run;
   Test_Gradient_Checking.Run;

   Test_Framework.Print_Summary;

   if not Test_Framework.All_Passed then
      Ada.Command_Line.Set_Exit_Status (Ada.Command_Line.Failure);
   end if;
end Test_Runner;
