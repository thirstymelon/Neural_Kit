with Ada.Numerics.Elementary_Functions;
with Ada.Text_IO;
with Digit_Data; use Digit_Data;
with Neural_Kit.Layer; use Neural_Kit.Layer;
with Neural_Kit.Random; use Neural_Kit.Random;
with Neural_Kit.Types; use Neural_Kit.Types;

procedure Mnist_Classifier is

   use Ada.Numerics.Elementary_Functions;
   use Ada.Text_IO;

   Gen : Generator;

   Hidden_Layer : Layer :=
     New_Layer
       (Input_Dimension  => 64,
        Output_Dimension => 40,
        Gen              => Gen,
        Activation       => ReLU,
        Init_Strategy    => He);

   Output_Layer : Layer :=
     New_Layer
       (Input_Dimension  => 40,
        Output_Dimension => 10,
        Gen              => Gen,
        Activation       => Softmax,
        Init_Strategy    => Xavier);

   Learning_Rate : constant Scalar := 0.035;
   Epoch_Count   : constant Positive := 250;

   Softmax_Grad : Vector (1 .. 10);
   Hidden_Grad  : Vector (1 .. 40);
   Dummy_Grad   : Vector (1 .. 64);

   Epoch_Loss : Scalar;

   function Shift_Sample
     (Pixels : Vector; Dr, Dc : Integer) return Vector
   is
      Result : Vector (1 .. 64) := (others => 0.0);
      Nr, Nc : Integer;
   begin
      for R in 0 .. 7 loop
         for C in 0 .. 7 loop
            Nr := R + Dr;
            Nc := C + Dc;
            if Nr in 0 .. 7 and then Nc in 0 .. 7 then
               Result (Nr * 8 + Nc + 1) := Pixels (R * 8 + C + 1);
            end if;
         end loop;
      end loop;
      return Result;
   end Shift_Sample;

   function Argmax (V : Vector) return Natural is
      Best_Idx : Positive := V'First;
      Best_Val : Scalar := V (Best_Idx);
   begin
      for I in V'First + 1 .. V'Last loop
         if V (I) > Best_Val then
            Best_Val := V (I);
            Best_Idx := I;
         end if;
      end loop;
      return Best_Idx - 1;
   end Argmax;

   procedure Print_Digit_ASCII (P : Vector) is
   begin
      for R in 0 .. 7 loop
         Put ("      ");
         for C in 0 .. 7 loop
            declare
               Idx : constant Positive := R * 8 + C + 1;
            begin
               if P (Idx) > 0.5 then
                  Put ("##");
               else
                  Put ("..");
               end if;
            end;
         end loop;
         New_Line;
      end loop;
   end Print_Digit_ASCII;

   function Bar_Graph (P : Scalar) return String is
      Bars : constant Natural := Natural (P * 20.0);
      Res  : String (1 .. 20) := (others => ' ');
   begin
      for I in 1 .. Bars loop
         Res (I) := '=';
      end loop;
      return "[" & Res & "] " & Integer'Image (Integer (P * 100.0)) & "%";
   end Bar_Graph;

begin

   Put_Line ("============================================================");
   Put_Line ("      NEURAL_KIT: HANDWRITTEN DIGIT RECOGNIZER (MNIST)      ");
   Put_Line ("     8x8 Grayscale Bitmap Classifier Built with Ada 2022    ");
   Put_Line ("============================================================");
   Put_Line ("");
   Put_Line
     ("Architecture: Input(64) -> Dense(40, ReLU) -> Dense(10, Softmax)");
   Put_Line ("Dataset:      8x8 Optical Recognition of Handwritten Digits");
   Put_Line
     ("Training set: 40 base samples + 8-way shift augmentation "
      & "(360 patterns)");
   Put_Line ("Test set:     20 samples (2 unseen variations per digit 0..9)");
   Put_Line ("Optimizer:    Analytical Layer Backpropagation (SGD)");
   Put_Line ("");

   Set_Seed (Gen, 12345);

   Put_Line ("Training neural network...");
   for Epoch in 1 .. Epoch_Count loop
      Epoch_Loss := 0.0;

      for S in 1 .. Train_Count loop
         declare
            Base_Sample : constant Digit_Sample := Training_Sample (S);
            Target      : constant Vector       := Base_Sample.One_Hot;

            procedure Train_One (Pix : Vector) is
            begin
               Hidden_Layer.Forward (Pix);
               Output_Layer.Forward (Hidden_Layer.Output);

               for C in 1 .. 10 loop
                  declare
                     Prob : constant Scalar :=
                       Scalar'Max (1.0e-7, Output_Layer.Output (C));
                  begin
                     if Target (C) > 0.5 then
                        Epoch_Loss := Epoch_Loss - Log (Prob);
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
                  Input_Activation => Pix,
                  Input_Grad       => Dummy_Grad);

               Output_Layer.Update (Learning_Rate);
               Hidden_Layer.Update (Learning_Rate);
            end Train_One;
         begin
            Train_One (Base_Sample.Pixels);
            for Dr in -1 .. 1 loop
               for Dc in -1 .. 1 loop
                  if Dr /= 0 or else Dc /= 0 then
                     Train_One (Shift_Sample (Base_Sample.Pixels, Dr, Dc));
                  end if;
               end loop;
            end loop;
         end;
      end loop;

      if Epoch mod 50 = 0 then
         Put_Line
           ("  Epoch "
            & Integer'Image (Epoch)
            & " /"
            & Integer'Image (Epoch_Count)
            & " | Mean Cross-Entropy Loss: "
            & Scalar'Image (Epoch_Loss / (Scalar (Train_Count) * 9.0)));
      end if;
   end loop;

   Put_Line ("");
   Put_Line ("============================================================");
   Put_Line ("             VISUAL SAMPLE DIGIT INFERENCE                  ");
   Put_Line ("============================================================");
   Put_Line ("");

   --  Show visual predictions for several test digits
   for S in 1 .. 6 loop
      declare
         Sample_Idx : constant Positive := S * 3;
         Test_Sample : constant Digit_Sample := Testing_Sample (Sample_Idx);
         Pred_Vec    : Vector (1 .. 10);
         Predicted_D : Natural;
         Is_Match    : Boolean;
      begin
         Hidden_Layer.Forward (Test_Sample.Pixels);
         Pred_Vec    := Output_Layer.Predict (Hidden_Layer.Output);
         Predicted_D := Argmax (Pred_Vec);
         Is_Match    := (Predicted_D = Test_Sample.True_Digit);

         Put_Line
           ("Test Sample #"
            & Integer'Image (Sample_Idx)
            & "  (Ground Truth: Digit "
            & Natural'Image (Test_Sample.True_Digit)
            & ")");
         Put_Line ("  Input 8x8 Bitmap:");
         Print_Digit_ASCII (Test_Sample.Pixels);
         Put_Line ("  Predicted Digit: "
                   & Natural'Image (Predicted_D)
                   & " ["
                   & (if Is_Match then "PASS" else "FAIL")
                   & "]");
         Put_Line ("  Confidence Distribution:");
         for D in 0 .. 9 loop
            if Pred_Vec (D + 1) >= 0.05 then
               Put_Line
                 ("    Digit "
                  & Natural'Image (D)
                  & ": "
                  & Bar_Graph (Pred_Vec (D + 1)));
            end if;
         end loop;
         Put_Line ("--------------------------------------------------");
      end;
   end loop;

   Put_Line ("");
   Put_Line ("============================================================");
   Put_Line ("              FULL TEST SET ACCURACY REPORT                 ");
   Put_Line ("============================================================");
   Put_Line ("");

   declare
      Correct_Count : Natural := 0;
   begin
      for S in 1 .. Test_Count loop
         declare
            Test_Sample : constant Digit_Sample := Testing_Sample (S);
            Pred_Vec    : Vector (1 .. 10);
            Predicted_D : Natural;
            Is_Correct  : Boolean;
         begin
            Hidden_Layer.Forward (Test_Sample.Pixels);
            Pred_Vec    := Output_Layer.Predict (Hidden_Layer.Output);
            Predicted_D := Argmax (Pred_Vec);
            Is_Correct  := (Predicted_D = Test_Sample.True_Digit);

            if Is_Correct then
               Correct_Count := Correct_Count + 1;
            end if;

            Put_Line
              ("Digit #"
               & (if S < 10 then " " else "")
               & Integer'Image (S)
               & " | Target: "
               & Natural'Image (Test_Sample.True_Digit)
               & " | Predicted: "
               & Natural'Image (Predicted_D)
               & " (Confidence: "
               & Integer'Image (Integer (Pred_Vec (Predicted_D + 1) * 100.0))
               & "%)"
               & (if Is_Correct then "  [OK]" else "  [MISMATCH]"));
         end;
      end loop;

      Put_Line ("");
      Put_Line ("--------------------------------------------------");
      Put_Line
        ("Final MNIST Test Accuracy: "
         & Integer'Image (Correct_Count)
         & " /"
         & Integer'Image (Test_Count)
         & " ("
         & Integer'Image (Integer
             ((Float (Correct_Count) / Float (Test_Count)) * 100.0))
         & " %)");
      Put_Line ("==================================================");
   end;

end Mnist_Classifier;
