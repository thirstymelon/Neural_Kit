with Ada.Text_IO;
with Neural_Kit.Neuron;
with Neural_Kit.Types;

procedure Example is

   use Ada.Text_IO;
   use Neural_Kit.Neuron;
   use Neural_Kit.Types;

   Neuron_1 : Neuron_T := New_Neuron (Weights => (0.5, 0.5), Bias => 0.5);

   Inputs : constant array (1 .. 4) of Features_T :=
     ((0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0));

   Targets : constant array (1 .. 4) of Scalar_T := (0.0, 0.0, 0.0, 1.0);

   Learning_Rate : constant Scalar_T := 0.1;
   Epoch_Count   : constant Positive := 10_000;

   Epoch_Loss : Scalar_T;

begin

   Put_Line ("Neural_Kit v0.1.0");
   Put_Line ("Single Neuron - AND");
   Put_Line ("");

   --  Training
   for Epoch in 1 .. Epoch_Count loop

      Epoch_Loss := 0.0;

      for I in Inputs'Range loop

         Neuron_1.Forward (Inputs (I));
         Neuron_1.Activate;

         Epoch_Loss := Epoch_Loss + Neuron_1.Loss (Targets (I));

         Neuron_1.Update
           (Input         => Inputs (I),
            Target        => Targets (I),
            Learning_Rate => Learning_Rate);

      end loop;

      Epoch_Loss := Epoch_Loss / Scalar_T (Inputs'Length);

      if Epoch mod 1_000 = 0 then

         Put_Line
           ("Epoch: "
            & Integer'Image (Epoch)
            & "  Loss: "
            & Scalar_T'Image (Epoch_Loss));

      end if;

   end loop;

   --  Testing
   Put_Line ("");
   Put_Line ("AND results:");
   Put_Line ("");

   for I in Inputs'Range loop

      declare
         Prediction : constant Scalar_T := Neuron_1.Predict (Inputs (I));

         Classification : constant Boolean := Prediction >= 0.5;

      begin

         Put_Line
           ("Input: ("
            & Scalar_T'Image (Inputs (I) (1))
            & ","
            & Scalar_T'Image (Inputs (I) (2))
            & ")"
            & "  Target: "
            & Scalar_T'Image (Targets (I))
            & "  Prediction: "
            & Scalar_T'Image (Prediction)
            & "  Class: "
            & Boolean'Image (Classification));

      end;

   end loop;

end Example;
