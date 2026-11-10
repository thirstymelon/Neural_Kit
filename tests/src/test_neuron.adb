with Ada.Assertions;
with Neural_Kit.Neuron; use Neural_Kit.Neuron;
with Neural_Kit.Types;  use Neural_Kit.Types;
with Test_Framework;    use Test_Framework;

package body Test_Neuron is

   procedure Run is
      Neuron_AND : Neuron :=
        New_Neuron (Weights => (0.5, 0.5), Bias => 0.5);

      type Training_Inputs is array (1 .. 4) of Vector (1 .. 2);
      Inputs : constant Training_Inputs :=
        ((0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0));

      Targets : constant array (1 .. 4) of Scalar :=
        (0.0, 0.0, 0.0, 1.0);

      Learning_Rate : constant Scalar := 0.2;
      Loss_Init     : Scalar := 0.0;
      Loss_Final    : Scalar := 0.0;
   begin
      Start_Suite ("Neuron Regression and Training");

      --  Initial loss computation
      for I in Inputs'Range loop
         declare
            Pred : constant Scalar := Neuron_AND.Predict (Inputs (I));
         begin
            Loss_Init := Loss_Init + Neuron_AND.Loss (Targets (I));
            pragma Unreferenced (Pred);
         end;
      end loop;
      Loss_Init := Loss_Init / 4.0;

      --  Train for 2,500 epochs
      for Epoch in 1 .. 2_500 loop
         for I in Inputs'Range loop
            Neuron_AND.Forward (Inputs (I));
            Neuron_AND.Activate;
            Neuron_AND.Update
              (Input         => Inputs (I),
               Target        => Targets (I),
               Learning_Rate => Learning_Rate);
         end loop;
      end loop;

      --  Final loss
      for I in Inputs'Range loop
         declare
            Pred : constant Scalar := Neuron_AND.Predict (Inputs (I));
         begin
            Loss_Final := Loss_Final + Neuron_AND.Loss (Targets (I));
            pragma Unreferenced (Pred);
         end;
      end loop;
      Loss_Final := Loss_Final / 4.0;

      Assert_True
        (Loss_Final < Loss_Init,
         Message => "Neuron training decreases loss: Initial="
                    & Scalar'Image (Loss_Init)
                    & ", Final="
                    & Scalar'Image (Loss_Final));

      Assert_True
        (Loss_Final < 0.05,
         Message => "Neuron reaches low loss on AND gate");

      --  Verify binary classifications
      Assert_True
        (not Neuron_AND.Classify_Binary ((0.0, 0.0)),
         Message => "AND (0,0) = False");
      Assert_True
        (not Neuron_AND.Classify_Binary ((0.0, 1.0)),
         Message => "AND (0,1) = False");
      Assert_True
        (not Neuron_AND.Classify_Binary ((1.0, 0.0)),
         Message => "AND (1,0) = False");
      Assert_True
        (Neuron_AND.Classify_Binary ((1.0, 1.0)),
         Message => "AND (1,1) = True");

      --  Arbitrary dimension: 4-input neuron
      declare
         N4 : Neuron :=
           New_Neuron
             (Weights => (0.1, -0.2, 0.3, 0.5),
              Bias    => 0.0);
         In4 : constant Vector (1 .. 4) := (1.0, 2.0, 3.0, 4.0);
         P4  : constant Scalar := N4.Predict (In4);
      begin
         Assert_True
           (P4 > 0.0 and then P4 < 1.0,
            Message => "4-dimensional neuron predict valid output");
      end;

      --  Dimension mismatch check on Neuron
      declare
         N_Dim2 : Neuron :=
           New_Neuron (Weights => (1.0, 2.0), Bias => 0.0);
         Bad_In : constant Vector (1 .. 3) := (1.0, 2.0, 3.0);
         Failed : Boolean := False;
      begin
         begin
            declare
               Dummy : constant Scalar := N_Dim2.Predict (Bad_In);
               pragma Unreferenced (Dummy);
            begin
               null;
            end;
         exception
            when Ada.Assertions.Assertion_Error | Dimension_Error =>
               Failed := True;
         end;
         Assert_True
           (Failed,
            Message => "Neuron forward dimension mismatch raises error");
      end;

      End_Suite;
   end Run;

end Test_Neuron;
