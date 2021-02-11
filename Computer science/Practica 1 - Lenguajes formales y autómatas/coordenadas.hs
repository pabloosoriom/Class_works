module Coords where
import Data.List as List


coords :: Int -> Int -> Int -> Int -> [Int]
coords x y n m
  | ((n == 1) && (m == 1)) = s
  | otherwise =
    if(x < l) then
      if(y < a) then
        s ++ [1] ++ (coords x y l a)
      else
        s ++ [0] ++ (coords x (y-l) l a)
    else
      if(y < l) then
        s ++ [3] ++ (coords (x-l) y l a)
      else
        s ++ [2] ++ (coords (x-l) (y-l) l a)
  where
    s = []
    l = (n `div` 2)
    a = (m `div` 2)