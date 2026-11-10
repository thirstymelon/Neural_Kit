with Ada.Numerics;
with Ada.Numerics.Elementary_Functions;

package body Neural_Kit.Random is

   use Ada.Numerics.Elementary_Functions;

   function Next_U32 (Gen : in out Generator) return Interfaces.Unsigned_32 is
      use Interfaces;
      Old_State   : constant Unsigned_64 := Gen.State;
      Multiplier  : constant Unsigned_64 := 16#5851F42D4C957F2D#;
      Xor_Shifted : Unsigned_32;
      Rot         : Natural;
   begin
      Gen.State := Old_State * Multiplier + Gen.Inc;
      Xor_Shifted :=
        Unsigned_32
          (Shift_Right
             ((Old_State xor Shift_Right (Old_State, 18)), 27)
           and 16#FFFF_FFFF#);
      Rot := Natural (Shift_Right (Old_State, 59));
      return Rotate_Right (Xor_Shifted, Rot);
   end Next_U32;

   function Random_Canonical (Gen : in out Generator) return Scalar is
      Raw : constant Interfaces.Unsigned_32 := Next_U32 (Gen);
   begin
      return Scalar (Raw) / 4294967296.0;
   end Random_Canonical;

   procedure Set_Seed
     (Gen  : in out Generator;
      Seed : Integer)
   is
      use Interfaces;
      Seed_64 : constant Unsigned_64 :=
        Unsigned_64 (Unsigned_32'Mod (Seed));
   begin
      Gen.State := Seed_64 xor 16#853C49E6748FEA9B#;
      Gen.Inc   := Shift_Left (Seed_64, 1) or 1;
      Gen.Has_Spare := False;
      Gen.Spare := 0.0;

      declare
         Discard : constant Unsigned_32 := Next_U32 (Gen);
         pragma Unreferenced (Discard);
      begin
         null;
      end;
   end Set_Seed;

   function Random_Uniform
     (Gen  : in out Generator;
      Low  : Scalar;
      High : Scalar) return Scalar
   is
   begin
      if Low > High then
         raise Numerical_Error with "Low must be less than or equal to High";
      end if;

      return Low + (High - Low) * Random_Canonical (Gen);
   end Random_Uniform;

   function Random_Normal
     (Gen     : in out Generator;
      Mean    : Scalar;
      Std_Dev : Scalar) return Scalar
   is
      U1  : Scalar;
      U2  : Scalar;
      Mag : Scalar;
      Z0  : Scalar;
      Z1  : Scalar;
   begin
      if Std_Dev < 0.0 then
         raise Numerical_Error with "Std_Dev must be non-negative";
      end if;

      if Std_Dev = 0.0 then
         return Mean;
      end if;

      if Gen.Has_Spare then
         Gen.Has_Spare := False;
         return Mean + Gen.Spare * Std_Dev;
      end if;

      loop
         U1 := Random_Canonical (Gen);
         exit when U1 > 0.0;
      end loop;

      U2 := Random_Canonical (Gen);

      Mag := Sqrt (-2.0 * Log (U1));
      Z0 := Mag * Cos (2.0 * Ada.Numerics.Pi * U2);
      Z1 := Mag * Sin (2.0 * Ada.Numerics.Pi * U2);

      Gen.Spare := Z1;
      Gen.Has_Spare := True;

      return Mean + Z0 * Std_Dev;
   end Random_Normal;

   function Init_Zeros (Dim : Positive) return Vector is
   begin
      return (1 .. Dim => 0.0);
   end Init_Zeros;

   function Init_Zeros (Rows, Cols : Positive) return Matrix is
   begin
      return (1 .. Rows => (1 .. Cols => 0.0));
   end Init_Zeros;

   function Init_Uniform
     (Gen       : in out Generator;
      Dim       : Positive;
      Low, High : Scalar) return Vector
   is
      Result : Vector (1 .. Dim);
   begin
      for I in 1 .. Dim loop
         Result (I) := Random_Uniform (Gen, Low, High);
      end loop;
      return Result;
   end Init_Uniform;

   function Init_Uniform
     (Gen        : in out Generator;
      Rows, Cols : Positive;
      Low, High  : Scalar) return Matrix
   is
      Result : Matrix (1 .. Rows, 1 .. Cols);
   begin
      for I in 1 .. Rows loop
         for J in 1 .. Cols loop
            Result (I, J) := Random_Uniform (Gen, Low, High);
         end loop;
      end loop;
      return Result;
   end Init_Uniform;

   function Init_Xavier
     (Gen        : in out Generator;
      Rows, Cols : Positive) return Matrix
   is
      Fan_In  : constant Scalar := Scalar (Cols);
      Fan_Out : constant Scalar := Scalar (Rows);
      Limit   : constant Scalar := Sqrt (6.0 / (Fan_In + Fan_Out));
   begin
      return Init_Uniform (Gen, Rows, Cols, -Limit, Limit);
   end Init_Xavier;

   function Init_He
     (Gen        : in out Generator;
      Rows, Cols : Positive) return Matrix
   is
      Fan_In  : constant Scalar := Scalar (Cols);
      Std_Dev : constant Scalar := Sqrt (2.0 / Fan_In);
      Result  : Matrix (1 .. Rows, 1 .. Cols);
   begin
      for I in 1 .. Rows loop
         for J in 1 .. Cols loop
            Result (I, J) := Random_Normal (Gen, 0.0, Std_Dev);
         end loop;
      end loop;
      return Result;
   end Init_He;

end Neural_Kit.Random;
