module Img2dfa where

import FiniteAutomata
import Data.Set as Set
import Data.Map as Map
import Data.List as List
import Codec.Picture

img2dfa :: ImagePixels -> FiniteAutomata ImagePixels Int
img2dfa img = aux 0 0 img img dfa
     where
     dfa :: FiniteAutomata ImagePixels Int
     dfa = FA {
        states = Set.singleton img
        , alphabet = imageAlphabet
        , delta        = delta'
        , initialState = img
        , acceptState  = Set.fromList [[[blackPx]]]
        } where delta'= Map.empty

scaleDiference :: ImagePixels -> Int
scaleDiference img
 | length img <=8 = 0
 | length img== 16= 1
 | length img== 32= 2
 | length img== 64= 3
 | length img== 128= 4
 | length img== 256= 5
 | length img== 528= 6
 | length img== 1024= 7
 | length img== 2048= 8


aux :: Int -> Int -> ImagePixels -> ImagePixels -> FiniteAutomata ImagePixels Int -> FiniteAutomata ImagePixels Int
aux i j img img2 dfa@(FA st sy tf as es) = if i==j && (i/=0&&j/=0)
   then dfa
 else u
    where
    dfa :: FiniteAutomata ImagePixels Int
    dfa = FA {
        states = Set.union st (Set.fromList imgP)
       ,alphabet = sy
       , delta  = if length img==1 then
                    addTransition (img, 0, img) $
                    addTransition (img, 1, img) $
                    addTransition (img, 2, img) $
                    addTransition (img, 3, img) $
                    tf
                    else
                    addTransition (img, 0, imgP !! 0) $
                    addTransition (img, 1, imgP !! 1) $
                    addTransition (img, 2, imgP !! 2) $
                    addTransition (img, 3, imgP !! 3) $
                    tf
      , initialState = as
      , acceptState  = Set.fromList [[[blackPx]]]
      }
    imgP = [splitList img k| k <- [0, 1, 2, 3]]
    i' = i + 1
    j' = j + (i-scaleDiference img2)
    fas = [aux i' j' image img2 dfa | image <- imgP]
    u = joinAu fas



joinAu :: [FiniteAutomata ImagePixels Int] -> FiniteAutomata ImagePixels Int
joinAu fas = FA{
      states = Set.unions (List.map states fas)
        , alphabet  = imageAlphabet
        , delta = delta1 (delta (head fas)) (tail (List.map delta (tail fas)))
        , initialState = initialState (head fas)
        , acceptState  = Set.unions (List.map acceptState fas)
      }
      where
        delta1 :: TransitionFunction ImagePixels Int -> [TransitionFunction ImagePixels Int] -> TransitionFunction ImagePixels Int
        delta1 tf [] = tf
        delta1 d tfs = Map.union d (delta1 (head tfs) (tail tfs))

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



return1Map :: FiniteAutomata ImagePixels Int -> ImagePixels -> Map Int (Set ImagePixels)
return1Map dfa img = (delta dfa)!img

returnPixel4 :: [Int] -> Map Int(Set ImagePixels) -> FiniteAutomata ImagePixels Int ->ImagePixels
returnPixel4 ad m dfa
  | length ad==1 = Set.elemAt 0 (m!(head ad))
  | otherwise = returnPixel4 (tail ad) (return1Map dfa (Set.elemAt 0 (m!(head ad)))) dfa

returnPixel2 :: Int -> Int -> FiniteAutomata ImagePixels Int -> Int -> PixelRGB8
returnPixel2 x y dfa n = ((returnPixel4 (coords x y n n) fMap dfa)!!0)!!0
 where
   fMap = return1Map dfa (initialState dfa)



createImage :: Int -> FiniteAutomata ImagePixels Int -> DynamicImage
createImage n dfa = ImageRGB8 (generateImage (\x y -> returnPixel2 x y dfa (2^n) ) (2^n) (2^n))



splitList ::  [[a]] -> Int -> [[a]]
splitList l a
  | length l == 1 = l
  | a==1 = [Prelude.take (length l `div` 2) x | x <- lT]
  | a==3 = [Prelude.drop (length l `div` 2) x | x <- lT]
  | a==0 = [Prelude.take (length l `div` 2) x | x <- lD]
  | a==2 = [Prelude.drop (length l `div` 2) x | x <- lD]
      where
       lT= Prelude.take (length l `div` 2) l
       lD= Prelude.drop (length l `div` 2) l


