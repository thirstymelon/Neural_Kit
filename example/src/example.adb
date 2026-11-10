with Ada.Text_IO;
with Neural_Kit.Layer;
with Neural_Kit.Neuron;
with Neural_Kit.Random;
with Neural_Kit.Types;

procedure Example is

   use Ada.Text_IO;
   use Neural_Kit.Layer;
   use Neural_Kit.Neuron;
   use Neural_Kit.Random;
   use Neural_Kit.Types;

begin

   Put_Line ("============================================================");
   Put_Line ("              NEURAL_KIT v0.2.0 DEMONSTRATION               ");
   Put_Line ("     Pure CPU Deep Learning from First Principles in Ada    ");
   Put_Line ("============================================================");
   Put_Line ("");

   ---------------------------------------------------------------------------
   --  Part 1: Single Neuron - Binary Classification & Gradient Descent
   ---------------------------------------------------------------------------
   Put_Line ("--- Part 1: Single Neuron Binary Classification (AND Gate) ---");
   Put_Line ("Learning linearly separable binary truth table via SGD.");
   Put_Line ("");

   declare
      N : Neuron := New_Neuron (Weights => (0.5, 0.5), Bias => 0.5);

      type Input_Table is array (1 .. 4) of Vector (1 .. 2);
      Inputs : constant Input_Table :=
        (((0.0, 0.0)),
         ((0.0, 1.0)),
         ((1.0, 0.0)),
         ((1.0, 1.0)));

      Targets : constant array (1 .. 4) of Scalar :=
        (0.0, 0.0, 0.0, 1.0);

      Learning_Rate : constant Scalar := 0.1;
      Epoch_Loss    : Scalar;
   begin
      for Epoch in 1 .. 10_000 loop
         Epoch_Loss := 0.0;
         for I in Inputs'Range loop
            N.Forward (Inputs (I));
            N.Activate;
            Epoch_Loss := Epoch_Loss + N.Loss (Targets (I));
            N.Update
              (Input         => Inputs (I),
               Target        => Targets (I),
               Learning_Rate => Learning_Rate);
         end loop;

         if Epoch mod 2_500 = 0 then
            Put_Line
              ("  Epoch "
               & Integer'Image (Epoch)
               & " | BCE Loss: "
               & Scalar'Image (Epoch_Loss / 4.0));
         end if;
      end loop;

      Put_Line ("");
      Put_Line ("  Trained AND Predictions:");
      for I in Inputs'Range loop
         declare
            Pred : constant Scalar := N.Predict (Inputs (I));
            Pass : constant Boolean := (Pred >= 0.5) = (Targets (I) >= 0.5);
         begin
            Put_Line
              ("    Input: ("
               & Scalar'Image (Inputs (I) (1))
               & ","
               & Scalar'Image (Inputs (I) (2))
               & ") -> Target: "
               & Scalar'Image (Targets (I))
               & " | Output: "
               & Scalar'Image (Pred)
               & " ["
               & (if Pass then "PASS" else "FAIL")
               & "]");
         end;
      end loop;
   end;

   Put_Line ("");

   ---------------------------------------------------------------------------
   --  Part 2: Multi-Neuron Dense Layer Forward Flow & Probability Modeling
   ---------------------------------------------------------------------------
   Put_Line ("--- Part 2: Dense Layer Forward Flow & Softmax Probability ---");
   Put_Line
     ("Architecture: Input(4) -> Hidden(4, ReLU) -> Output(3, Softmax)");
   Put_Line ("");

   declare
      Gen : Generator;

      Hidden : Layer :=
        New_Layer
          (Input_Dimension  => 4,
           Output_Dimension => 4,
           Gen              => Gen,
           Activation       => ReLU,
           Init_Strategy    => He);

      Classifier : Layer :=
        New_Layer
          (Input_Dimension  => 4,
           Output_Dimension => 3,
           Gen              => Gen,
           Activation       => Softmax,
           Init_Strategy    => Xavier);

      Sample_Features : constant Vector (1 .. 4) :=
        (0.8, -0.5, 1.2, -0.1);

      Hidden_Output : constant Vector :=
        Hidden.Predict (Sample_Features);

      Probabilities : constant Vector :=
        Classifier.Predict (Hidden_Output);

      Sum_Prob : Scalar := 0.0;
   begin
      Set_Seed (Gen, 42);

      Put_Line
        ("  Input Features (4D): ("
         & Scalar'Image (Sample_Features (1)) & ","
         & Scalar'Image (Sample_Features (2)) & ","
         & Scalar'Image (Sample_Features (3)) & ","
         & Scalar'Image (Sample_Features (4)) & ")");

      Put_Line
        ("  Hidden Activations (ReLU, 4 neurons): ("
         & Scalar'Image (Hidden_Output (1)) & ","
         & Scalar'Image (Hidden_Output (2)) & ","
         & Scalar'Image (Hidden_Output (3)) & ","
         & Scalar'Image (Hidden_Output (4)) & ")");

      Put_Line
        ("  Class Probabilities (Softmax, 3 classes): ("
         & Scalar'Image (Probabilities (1)) & ","
         & Scalar'Image (Probabilities (2)) & ","
         & Scalar'Image (Probabilities (3)) & ")");

      for I in Probabilities'Range loop
         Sum_Prob := Sum_Prob + Probabilities (I);
      end loop;

      Put_Line
        ("  Probability Sum (Normalized): "
         & Scalar'Image (Sum_Prob));
   end;

   Put_Line ("");

   ---------------------------------------------------------------------------
   --  Part 3: End-to-End Two-Layer Network Trained with Backpropagation
   ---------------------------------------------------------------------------
   Put_Line ("--- Part 3: Deep 2-Layer Network Solving XOR Non-Linearity ---");
   Put_Line ("Architecture: Input(2) -> Dense(4, Tanh) -> Dense(1, Sigmoid)");
   Put_Line ("Trained using pure v0.2.0 Layer analytical backpropagation.");
   Put_Line ("");

   declare
      Gen : Generator;

      L1 : Layer :=
        New_Layer
          (Input_Dimension  => 2,
           Output_Dimension => 4,
           Gen              => Gen,
           Activation       => Tanh,
           Init_Strategy    => Xavier);

      L2 : Layer :=
        New_Layer
          (Input_Dimension  => 4,
           Output_Dimension => 1,
           Gen              => Gen,
           Activation       => Sigmoid,
           Init_Strategy    => Xavier);

      type XOR_Inputs  is array (1 .. 4) of Vector (1 .. 2);
      type XOR_Targets is array (1 .. 4) of Vector (1 .. 1);

      X_Train : constant XOR_Inputs :=
        (((0.0, 0.0)),
         ((0.0, 1.0)),
         ((1.0, 0.0)),
         ((1.0, 1.0)));

      Y_Train : constant XOR_Targets :=
        (((1 => 0.0)),
         ((1 => 1.0)),
         ((1 => 1.0)),
         ((1 => 0.0)));

      Learning_Rate : constant Scalar := 0.6;
      Epoch_Loss    : Scalar;

      Out_Grad    : Vector (1 .. 1);
      Hidden_Grad : Vector (1 .. 4);
      Dummy_Grad  : Vector (1 .. 2);
   begin
      Set_Seed (Gen, 777);

      for Epoch in 1 .. 3_000 loop
         Epoch_Loss := 0.0;

         for S in 1 .. 4 loop
            L1.Forward (X_Train (S));
            L2.Forward (L1.Output);

            declare
               Pred : constant Scalar := L2.Output (1);
               Targ : constant Scalar := Y_Train (S) (1);
               Diff : constant Scalar := Pred - Targ;
            begin
               Epoch_Loss := Epoch_Loss + Diff * Diff;
               Out_Grad (1) := 2.0 * Diff;
            end;

            L2.Backward
              (Output_Grad      => Out_Grad,
               Input_Activation => L1.Output,
               Input_Grad       => Hidden_Grad);

            L1.Backward
              (Output_Grad      => Hidden_Grad,
               Input_Activation => X_Train (S),
               Input_Grad       => Dummy_Grad);

            L2.Update (Learning_Rate);
            L1.Update (Learning_Rate);
         end loop;

         if Epoch mod 1_000 = 0 then
            Put_Line
              ("  Epoch "
               & Integer'Image (Epoch)
               & " | MSE Loss: "
               & Scalar'Image (Epoch_Loss / 4.0));
         end if;
      end loop;

      Put_Line ("");
      Put_Line ("  Final Non-Linear XOR Predictions:");
      for S in 1 .. 4 loop
         L1.Forward (X_Train (S));
         L2.Forward (L1.Output);

         declare
            Pred : constant Scalar := L2.Output (1);
            Targ : constant Scalar := Y_Train (S) (1);
            Pass : constant Boolean := (Pred >= 0.5) = (Targ >= 0.5);
         begin
            Put_Line
              ("    Input: ("
               & Scalar'Image (X_Train (S) (1))
               & ","
               & Scalar'Image (X_Train (S) (2))
               & ") -> Target: "
               & Scalar'Image (Targ)
               & " | Predicted: "
               & Scalar'Image (Pred)
               & " ["
               & (if Pass then "PASS" else "FAIL")
               & "]");
         end;
      end loop;
   end;

   Put_Line ("");
   Put_Line ("============================================================");
   Put_Line ("          All demonstrations successfully completed!         ");
   Put_Line ("============================================================");

end Example;
