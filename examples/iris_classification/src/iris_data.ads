with Neural_Kit.Types; use Neural_Kit.Types;

package Iris_Data is

   type Feature_Array is array (1 .. 4) of Scalar;
   type Class_Kind is (Setosa, Versicolor, Virginica);

   type Sample is record
      Features   : Vector (1 .. 4);
      One_Hot    : Vector (1 .. 3);
      True_Class : Class_Kind;
   end record;

   Train_Count : constant Positive := 36;
   Test_Count  : constant Positive := 15;

   type Train_Set is array (1 .. Train_Count) of Sample;
   type Test_Set is array (1 .. Test_Count) of Sample;

   Training_Samples : constant Train_Set :=
     (
      --  Setosa samples
      (Features   => (-0.9, 1.0, -1.3, -1.3),
       One_Hot    => (1.0, 0.0, 0.0),
       True_Class => Setosa),
      (Features   => (-1.1, -0.1, -1.3, -1.3),
       One_Hot    => (1.0, 0.0, 0.0),
       True_Class => Setosa),
      (Features   => (-1.4, 0.3, -1.4, -1.3),
       One_Hot    => (1.0, 0.0, 0.0),
       True_Class => Setosa),
      (Features   => (-1.5, 0.1, -1.3, -1.3),
       One_Hot    => (1.0, 0.0, 0.0),
       True_Class => Setosa),
      (Features   => (-1.0, 1.2, -1.3, -1.3),
       One_Hot    => (1.0, 0.0, 0.0),
       True_Class => Setosa),
      (Features   => (-0.5, 1.9, -1.2, -1.1),
       One_Hot    => (1.0, 0.0, 0.0),
       True_Class => Setosa),
      (Features   => (-1.4, 0.8, -1.3, -1.2),
       One_Hot    => (1.0, 0.0, 0.0),
       True_Class => Setosa),
      (Features   => (-1.0, 0.8, -1.3, -1.3),
       One_Hot    => (1.0, 0.0, 0.0),
       True_Class => Setosa),
      (Features   => (-1.7, -0.4, -1.3, -1.3),
       One_Hot    => (1.0, 0.0, 0.0),
       True_Class => Setosa),
      (Features   => (-1.1, 0.1, -1.3, -1.4),
       One_Hot    => (1.0, 0.0, 0.0),
       True_Class => Setosa),
      (Features   => (-0.5, 1.5, -1.3, -1.3),
       One_Hot    => (1.0, 0.0, 0.0),
       True_Class => Setosa),
      (Features   => (-1.2, 0.8, -1.2, -1.3),
       One_Hot    => (1.0, 0.0, 0.0),
       True_Class => Setosa),

      --  Versicolor samples
      (Features   => (1.4, 0.3, 0.5, 0.3),
       One_Hot    => (0.0, 1.0, 0.0),
       True_Class => Versicolor),
      (Features   => (0.7, 0.3, 0.4, 0.4),
       One_Hot    => (0.0, 1.0, 0.0),
       True_Class => Versicolor),
      (Features   => (1.3, 0.1, 0.6, 0.4),
       One_Hot    => (0.0, 1.0, 0.0),
       True_Class => Versicolor),
      (Features   => (-0.4, -1.7, 0.1, 0.1),
       One_Hot    => (0.0, 1.0, 0.0),
       True_Class => Versicolor),
      (Features   => (0.8, -0.6, 0.5, 0.4),
       One_Hot    => (0.0, 1.0, 0.0),
       True_Class => Versicolor),
      (Features   => (-0.1, -0.6, 0.4, 0.1),
       One_Hot    => (0.0, 1.0, 0.0),
       True_Class => Versicolor),
      (Features   => (0.6, 0.6, 0.5, 0.5),
       One_Hot    => (0.0, 1.0, 0.0),
       True_Class => Versicolor),
      (Features   => (-1.1, -1.5, -0.3, -0.3),
       One_Hot    => (0.0, 1.0, 0.0),
       True_Class => Versicolor),
      (Features   => (-0.8, -0.8, 0.1, 0.0),
       One_Hot    => (0.0, 1.0, 0.0),
       True_Class => Versicolor),
      (Features   => (0.3, -0.8, 0.1, 0.0),
       One_Hot    => (0.0, 1.0, 0.0),
       True_Class => Versicolor),
      (Features   => (-0.9, -2.4, -0.3, -0.3),
       One_Hot    => (0.0, 1.0, 0.0),
       True_Class => Versicolor),
      (Features   => (0.3, -0.1, 0.5, 0.4),
       One_Hot    => (0.0, 1.0, 0.0),
       True_Class => Versicolor),

      --  Virginica samples
      (Features   => (0.6, 0.6, 1.3, 1.7),
       One_Hot    => (0.0, 0.0, 1.0),
       True_Class => Virginica),
      (Features   => (-0.1, -0.8, 0.8, 0.9),
       One_Hot    => (0.0, 0.0, 1.0),
       True_Class => Virginica),
      (Features   => (1.5, 0.1, 1.2, 1.2),
       One_Hot    => (0.0, 0.0, 1.0),
       True_Class => Virginica),
      (Features   => (0.6, -0.4, 1.0, 0.8),
       One_Hot    => (0.0, 0.0, 1.0),
       True_Class => Virginica),
      (Features   => (0.8, -0.1, 1.2, 1.3),
       One_Hot    => (0.0, 0.0, 1.0),
       True_Class => Virginica),
      (Features   => (2.1, -0.1, 1.6, 1.2),
       One_Hot    => (0.0, 0.0, 1.0),
       True_Class => Virginica),
      (Features   => (-1.1, -1.3, 0.4, 0.7),
       One_Hot    => (0.0, 0.0, 1.0),
       True_Class => Virginica),
      (Features   => (1.8, 1.9, 1.5, 1.7),
       One_Hot    => (0.0, 0.0, 1.0),
       True_Class => Virginica),
      (Features   => (1.0, -1.3, 1.1, 0.8),
       One_Hot    => (0.0, 0.0, 1.0),
       True_Class => Virginica),
      (Features   => (1.7, 1.2, 1.7, 1.7),
       One_Hot    => (0.0, 0.0, 1.0),
       True_Class => Virginica),
      (Features   => (0.8, 0.3, 0.8, 1.4),
       One_Hot    => (0.0, 0.0, 1.0),
       True_Class => Virginica),
      (Features   => (1.3, 0.3, 1.2, 1.5),
       One_Hot    => (0.0, 0.0, 1.0),
       True_Class => Virginica)
     );

   Testing_Samples : constant Test_Set :=
     (
      --  Test Setosa
      (Features   => (-1.6, 0.1, -1.4, -1.3),
       One_Hot    => (1.0, 0.0, 0.0),
       True_Class => Setosa),
      (Features   => (-0.7, 0.8, -1.3, -1.3),
       One_Hot    => (1.0, 0.0, 0.0),
       True_Class => Setosa),
      (Features   => (-1.3, 0.6, -1.3, -1.4),
       One_Hot    => (1.0, 0.0, 0.0),
       True_Class => Setosa),
      (Features   => (-1.0, 1.7, -1.2, -1.2),
       One_Hot    => (1.0, 0.0, 0.0),
       True_Class => Setosa),
      (Features   => (-0.6, 1.0, -1.3, -1.3),
       One_Hot    => (1.0, 0.0, 0.0),
       True_Class => Setosa),

      --  Test Versicolor
      (Features   => (1.2, -0.6, 0.6, 0.4),
       One_Hot    => (0.0, 1.0, 0.0),
       True_Class => Versicolor),
      (Features   => (0.3, -1.5, 0.4, 0.1),
       One_Hot    => (0.0, 1.0, 0.0),
       True_Class => Versicolor),
      (Features   => (-0.2, -0.8, 0.3, 0.1),
       One_Hot    => (0.0, 1.0, 0.0),
       True_Class => Versicolor),
      (Features   => (0.9, -0.1, 0.5, 0.4),
       One_Hot    => (0.0, 1.0, 0.0),
       True_Class => Versicolor),
      (Features   => (0.1, -0.4, 0.4, 0.1),
       One_Hot    => (0.0, 1.0, 0.0),
       True_Class => Versicolor),

      --  Test Virginica
      (Features   => (0.7, 0.3, 1.1, 1.7),
       One_Hot    => (0.0, 0.0, 1.0),
       True_Class => Virginica),
      (Features   => (1.9, 0.8, 1.5, 1.1),
       One_Hot    => (0.0, 0.0, 1.0),
       True_Class => Virginica),
      (Features   => (1.7, 0.1, 1.4, 1.4),
       One_Hot    => (0.0, 0.0, 1.0),
       True_Class => Virginica),
      (Features   => (1.1, 0.8, 1.2, 1.7),
       One_Hot    => (0.0, 0.0, 1.0),
       True_Class => Virginica),
      (Features   => (0.4, -0.6, 1.0, 1.4),
       One_Hot    => (0.0, 0.0, 1.0),
       True_Class => Virginica)
     );

end Iris_Data;
