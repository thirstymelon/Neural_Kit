with Interfaces;
with Neural_Kit.Types; use Neural_Kit.Types;

package Neural_Kit.Random is

   type Generator is limited private;

   procedure Set_Seed
     (Gen  : in out Generator;
      Seed : Integer);

   function Random_Uniform
     (Gen  : in out Generator;
      Low  : Scalar;
      High : Scalar) return Scalar
     with Pre => Low <= High;

   function Random_Normal
     (Gen     : in out Generator;
      Mean    : Scalar;
      Std_Dev : Scalar) return Scalar
     with Pre => Std_Dev >= 0.0;

   function Init_Zeros (Dim : Positive) return Vector
     with Post => Init_Zeros'Result'Length = Dim
          and then Init_Zeros'Result'First = 1;

   function Init_Zeros (Rows, Cols : Positive) return Matrix
     with Post => Init_Zeros'Result'Length (1) = Rows
          and then Init_Zeros'Result'Length (2) = Cols
          and then Init_Zeros'Result'First (1) = 1
          and then Init_Zeros'Result'First (2) = 1;

   function Init_Uniform
     (Gen       : in out Generator;
      Dim       : Positive;
      Low, High : Scalar) return Vector
     with Pre  => Low <= High,
          Post => Init_Uniform'Result'Length = Dim
          and then Init_Uniform'Result'First = 1;

   function Init_Uniform
     (Gen        : in out Generator;
      Rows, Cols : Positive;
      Low, High  : Scalar) return Matrix
     with Pre  => Low <= High,
          Post => Init_Uniform'Result'Length (1) = Rows
          and then Init_Uniform'Result'Length (2) = Cols
          and then Init_Uniform'Result'First (1) = 1
          and then Init_Uniform'Result'First (2) = 1;

   function Init_Xavier
     (Gen        : in out Generator;
      Rows, Cols : Positive) return Matrix
     with Post => Init_Xavier'Result'Length (1) = Rows
          and then Init_Xavier'Result'Length (2) = Cols
          and then Init_Xavier'Result'First (1) = 1
          and then Init_Xavier'Result'First (2) = 1;

   function Init_He
     (Gen        : in out Generator;
      Rows, Cols : Positive) return Matrix
     with Post => Init_He'Result'Length (1) = Rows
          and then Init_He'Result'Length (2) = Cols
          and then Init_He'Result'First (1) = 1
          and then Init_He'Result'First (2) = 1;

private

   type Generator is limited record
      State     : Interfaces.Unsigned_64 := 16#853C49E6748FEA9B#;
      Inc       : Interfaces.Unsigned_64 := 16#DA3E39CB94B95BDB#;
      Has_Spare : Boolean                := False;
      Spare     : Scalar                 := 0.0;
   end record;

end Neural_Kit.Random;
