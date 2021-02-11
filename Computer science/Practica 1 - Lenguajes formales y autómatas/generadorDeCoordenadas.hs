module GenerateCoords where
import Data.List as List

generateCoords :: Int -> [(Int,Int)]
generateCoords n
    | n>0 = cartProd [0..(n-1)] [0..(n-1)]
    | otherwise = []

cartProd :: [Int] -> [Int] -> [(Int,Int)]
cartProd xs ys = [(x,y) | x <- xs, y <- ys]
