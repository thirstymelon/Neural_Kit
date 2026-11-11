with Ada.Numerics.Elementary_Functions;
with Ada.Text_IO;
with Iris_Data; use Iris_Data;
with Neural_Kit.Layer; use Neural_Kit.Layer;
with Neural_Kit.Random; use Neural_Kit.Random;
with Neural_Kit.Types; use Neural_Kit.Types;

procedure Iris_Classification is

   use Ada.Numerics.Elementary_Functions;
   use Ada.Text_IO;

   Gen : Generator;

   Hidden_Layer : Layer :=
     New_Layer
       (Input_Dimension  => 4,
        Output_Dimension => 8,
        Gen              => Gen,
        Activation       => ReLU,
        Init_Strategy    => He);

   Output_Layer : Layer :=
     New_Layer
       (Input_Dimension  => 8,
        Output_Dimension => 3,
        Gen              => Gen,
        Activation       => Softmax,
        Init_Strategy    => Xavier);

   Learning_Rate : constant Scalar := 0.05;
   Epoch_Count   : constant Positive := 600;

   Softmax_Grad : Vector (1 .. 3);
   Hidden_Grad  : Vector (1 .. 8);
   Dummy_Grad   : Vector (1 .. 4);

   Epoch_Loss : Scalar;

   function Argmax (V : Vector) return Class_Kind is
      Best_Idx : Positive := V'First;
      Best_Val : Scalar := V (Best_Idx);
   begin
      for I in V'First + 1 .. V'Last loop
         if V (I) > Best_Val then
            Best_Val := V (I);
            Best_Idx := I;
         end if;
      end loop;

      case Best_Idx is
         when 1 => return Setosa;
         when 2 => return Versicolor;
         when others => return Virginica;
      end case;
   end Argmax;

   function Format_Pct (Val : Scalar) return String is
      Pct : constant Integer := Integer (Val * 100.0);
   begin
      if Pct < 10 then
         return " " & Integer'Image (Pct) & "%";
      else
         return Integer'Image (Pct) & "%";
      end if;
   end Format_Pct;

begin

   Put_Line ("============================================================");
   Put_Line ("      NEURAL_KIT: IRIS FLOWER SPECIES CLASSIFIER            ");
   Put_Line ("    Multi-Class Neural Network Trained from First Principles");
   Put_Line ("============================================================");
   Put_Line ("");
   Put_Line ("Architecture: Input(4) -> Dense(8, ReLU) -> Dense(3, Softmax)");
   Put_Line ("Dataset:      Fisher's Iris Plant Database (Fisher, 1936)");
   Put_Line ("Training set: 36 samples (12 per species)");
   Put_Line ("Test set:     15 samples (5 per species)");
   Put_Line ("Optimizer:    Analytical Layer Backpropagation (SGD)");
   Put_Line ("");

   Set_Seed (Gen, 42);

   Put_Line ("Training progress:");
   for Epoch in 1 .. Epoch_Count loop
      Epoch_Loss := 0.0;

      for S in 1 .. Train_Count loop
         declare
            Sample_In : constant Vector := Training_Samples (S).Features;
            Target    : constant Vector := Training_Samples (S).One_Hot;
         begin
            Hidden_Layer.Forward (Sample_In);
            Output_Layer.Forward (Hidden_Layer.Output);

            for C in 1 .. 3 loop
               declare
                  P : constant Scalar :=
                    Scalar'Max (1.0e-7, Output_Layer.Output (C));
               begin
                  if Target (C) > 0.5 then
                     Epoch_Loss := Epoch_Loss - Log (P);
                  end if;
               end;
               Softmax_Grad (C) := Output_Layer.Output (C) - Target (C);
            end loop;

            Output_Layer.Backward
              (Output_Grad      => Softmax_Grad,
               Input_Activation => Hidden_Layer.Output,
               Input_Grad       => Hidden_Grad);

            Hidden_Layer.Backward
              (Output_Grad      => Hidden_Grad,
               Input_Activation => Sample_In,
               Input_Grad       => Dummy_Grad);

            Output_Layer.Update (Learning_Rate);
            Hidden_Layer.Update (Learning_Rate);
         end;
      end loop;

      if Epoch mod 100 = 0 then
         Put_Line
           ("  Epoch "
            & Integer'Image (Epoch)
            & " /"
            & Integer'Image (Epoch_Count)
            & " | Cross-Entropy Loss: "
            & Scalar'Image (Epoch_Loss / Scalar (Train_Count)));
      end if;
   end loop;

   Put_Line ("");
   Put_Line ("============================================================");
   Put_Line ("                   TEST SET EVALUATION                      ");
   Put_Line ("============================================================");
   Put_Line ("");

   declare
      Correct_Count : Natural := 0;
      Confusion     : array (Class_Kind, Class_Kind) of Natural :=
        (others => (others => 0));
   begin
      for S in 1 .. Test_Count loop
         declare
            Sample_In  : constant Vector := Testing_Samples (S).Features;
            True_C     : constant Class_Kind := Testing_Samples (S).True_Class;
            Pred_Vec   : Vector (1 .. 3);
            Pred_C     : Class_Kind;
            Is_Correct : Boolean;
         begin
            Hidden_Layer.Forward (Sample_In);
            Pred_Vec := Output_Layer.Predict (Hidden_Layer.Output);
            Pred_C   := Argmax (Pred_Vec);
            Is_Correct := (Pred_C = True_C);

            if Is_Correct then
               Correct_Count := Correct_Count + 1;
            end if;

            Confusion (True_C, Pred_C) := Confusion (True_C, Pred_C) + 1;

            Put_Line
              ("Sample "
               & (if S < 10 then " " else "")
               & Integer'Image (S)
               & " | True: "
               & Class_Kind'Image (True_C)
               & (case True_C is
                     when Setosa => "    ",
                     when Versicolor => "",
                     when Virginica => " ")
               & " | Pred: "
               & Class_Kind'Image (Pred_C)
               & (case Pred_C is
                     when Setosa => "    ",
                     when Versicolor => "",
                     when Virginica => " ")
               & " ["
               & Format_Pct (Pred_Vec (1)) & " /"
               & Format_Pct (Pred_Vec (2)) & " /"
               & Format_Pct (Pred_Vec (3)) & "]"
               & (if Is_Correct then "  [OK]" else "  [MISMATCH]"));
         end;
      end loop;

      Put_Line ("");
      Put_Line ("--------------------------------------------------");
      Put_Line ("CONFUSION MATRIX:");
      Put_Line ("                 Predicted:  Setosa  Versicolor  Virginica");
      Put_Line ("Actual Setosa:                   "
                & Integer'Image (Confusion (Setosa, Setosa))
                & "          "
                & Integer'Image (Confusion (Setosa, Versicolor))
                & "          "
                & Integer'Image (Confusion (Setosa, Virginica)));
      Put_Line ("Actual Versicolor:               "
                & Integer'Image (Confusion (Versicolor, Setosa))
                & "          "
                & Integer'Image (Confusion (Versicolor, Versicolor))
                & "          "
                & Integer'Image (Confusion (Versicolor, Virginica)));
      Put_Line ("Actual Virginica:                "
                & Integer'Image (Confusion (Virginica, Setosa))
                & "          "
                & Integer'Image (Confusion (Virginica, Versicolor))
                & "          "
                & Integer'Image (Confusion (Virginica, Virginica)));
      Put_Line ("--------------------------------------------------");
      Put_Line
        ("Final Test Accuracy: "
         & Integer'Image (Correct_Count)
         & " /"
         & Integer'Image (Test_Count)
         & " ("
         & Integer'Image (Integer
             ((Float (Correct_Count) / Float (Test_Count)) * 100.0))
         & " %)");
      Put_Line ("==================================================");
   end;

end Iris_Classification;
