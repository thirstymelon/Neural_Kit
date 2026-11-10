with Ada.Numerics.Elementary_Functions;
with Neural_Kit.Random; use Neural_Kit.Random;
with Neural_Kit.Types;  use Neural_Kit.Types;
with Test_Framework;    use Test_Framework;

package body Test_Random is

   use Ada.Numerics.Elementary_Functions;

   procedure Run is
      Gen1 : Generator;
      Gen2 : Generator;
      Gen3 : Generator;
   begin
      Start_Suite ("Random and Initialization");

      --  Reproducibility with identical seeds
      Set_Seed (Gen1, 1337);
      Set_Seed (Gen2, 1337);

      declare
         All_Match : Boolean := True;
      begin
         for I in 1 .. 20 loop
            declare
               V1 : constant Scalar := Random_Uniform (Gen1, -1.0, 1.0);
               V2 : constant Scalar := Random_Uniform (Gen2, -1.0, 1.0);
            begin
               if V1 /= V2 then
                  All_Match := False;
                  exit;
               end if;
            end;
         end loop;
         Assert_True
           (All_Match,
            Message => "Identical seeds produce identical sequence");
      end;

      --  Different sequences with different seeds
      Set_Seed (Gen3, 42);
      declare
         Different : Boolean := False;
      begin
         for I in 1 .. 10 loop
            declare
               V1 : constant Scalar := Random_Uniform (Gen1, 0.0, 1.0);
               V3 : constant Scalar := Random_Uniform (Gen3, 0.0, 1.0);
            begin
               if V1 /= V3 then
                  Different := True;
                  exit;
               end if;
            end;
         end loop;
         Assert_True
           (Different,
            Message => "Different seeds produce different sequence");
      end;

      --  Uniform distribution bounds
      declare
         Within_Bounds : Boolean := True;
      begin
         for I in 1 .. 100 loop
            declare
               Val : constant Scalar :=
                 Random_Uniform (Gen1, 5.0, 10.0);
            begin
               if Val < 5.0 or else Val >= 10.0 then
                  Within_Bounds := False;
                  exit;
               end if;
            end;
         end loop;
         Assert_True
           (Within_Bounds,
            Message => "Uniform random values strictly in [5.0, 10.0)");
      end;

      --  Normal distribution sample mean and variance
      declare
         N_Samples : constant Positive := 1_000;
         Sum_Val   : Scalar := 0.0;
         Sum_Sq    : Scalar := 0.0;
         Mean_Val  : Scalar;
         Var_Val   : Scalar;
      begin
         for I in 1 .. N_Samples loop
            declare
               X : constant Scalar := Random_Normal (Gen1, 0.0, 1.0);
            begin
               Sum_Val := Sum_Val + X;
               Sum_Sq := Sum_Sq + X ** 2;
            end;
         end loop;
         Mean_Val := Sum_Val / Scalar (N_Samples);
         Var_Val := (Sum_Sq / Scalar (N_Samples)) - Mean_Val ** 2;

         --  Sample mean should be near 0 (within +/- 0.15)
         Assert_True
           (abs (Mean_Val) < 0.15,
            Message => "Normal(0,1) sample mean near 0: "
                       & Scalar'Image (Mean_Val));

         --  Sample variance should be near 1.0 (within 0.8 .. 1.2)
         Assert_True
           (Var_Val > 0.8 and then Var_Val < 1.2,
            Message => "Normal(0,1) sample variance near 1: "
                       & Scalar'Image (Var_Val));
      end;

      --  Zero Initializations
      declare
         Z_Vec : constant Vector := Init_Zeros (5);
         Z_Mat : constant Matrix := Init_Zeros (3, 4);
         All_Zero : Boolean := True;
      begin
         for I in Z_Vec'Range loop
            if Z_Vec (I) /= 0.0 then
               All_Zero := False;
            end if;
         end loop;
         for I in Z_Mat'Range (1) loop
            for J in Z_Mat'Range (2) loop
               if Z_Mat (I, J) /= 0.0 then
                  All_Zero := False;
               end if;
            end loop;
         end loop;
         Assert_True (All_Zero, Message => "Init_Zeros produces zeroed arrays");
         Assert_True
           (Z_Vec'Length = 5 and then Z_Mat'Length (1) = 3
            and then Z_Mat'Length (2) = 4,
            Message => "Init_Zeros correct dimensions");
      end;

      --  Xavier Initialization Bounds
      declare
         Rows    : constant Positive := 10;
         Cols    : constant Positive := 20;
         Limit   : constant Scalar :=
           Sqrt (6.0 / (Scalar (Cols) + Scalar (Rows)));
         X_Mat   : constant Matrix := Init_Xavier (Gen1, Rows, Cols);
         Bounded : Boolean := True;
      begin
         for I in X_Mat'Range (1) loop
            for J in X_Mat'Range (2) loop
               if X_Mat (I, J) < -Limit or else X_Mat (I, J) > Limit then
                  Bounded := False;
               end if;
            end loop;
         end loop;
         Assert_True
           (Bounded,
            Message => "Xavier weights stay within [-Limit, +Limit]");
      end;

      --  He Initialization Dimensions and Non-Zero
      declare
         H_Mat : constant Matrix := Init_He (Gen1, 8, 16);
         Has_Non_Zero : Boolean := False;
      begin
         for I in H_Mat'Range (1) loop
            for J in H_Mat'Range (2) loop
               if H_Mat (I, J) /= 0.0 then
                  Has_Non_Zero := True;
               end if;
            end loop;
         end loop;
         Assert_True
           (Has_Non_Zero and then H_Mat'Length (1) = 8
            and then H_Mat'Length (2) = 16,
            Message => "Init_He produces non-zero matrix of correct shape");
      end;

      End_Suite;
   end Run;

end Test_Random;

