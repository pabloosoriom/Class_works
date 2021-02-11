module Img2dfa where

import FiniteAutomata
import Data.Set as Set
import Data.Map as Map
import Data.List as List

img2dfa :: ImagePixels -> FiniteAutomata ImagePixels Int
img2dfa img = aux 0 0 img dfa 
     where 
     dfa :: FiniteAutomata ImagePixels Int
     dfa = FA {
        states = Set.singleton img
        , alphabet = imageAlphabet 
        , delta        = delta'
        , initialState = img
        , acceptState  = Set.fromList [[[blackPx]]]
        } where delta'= Map.empty

aux :: Int -> Int -> ImagePixels -> FiniteAutomata ImagePixels Int -> FiniteAutomata ImagePixels Int
aux i j img dfa@(FA st sy tf as es) = if i == j then dfa else u 
    where 
    dfa :: FiniteAutomata ImagePixels Int
    dfa = FA {
        states = Set.union st (Set.fromList imgP)
       ,alphabet = sy
       , delta  = addTransition (img, 0, imgP !! 0) $  
                  addTransition (img, 1, imgP !! 1) $ 
                  addTransition (img, 2, imgP !! 2) $ 
                  addTransition (img, 3, imgP !! 3) $
                  tf  
      , initialState = as 
      , acceptState  = Set.fromList [[[blackPx]]]
      } 
    imgP = [splitImg k img| k <- [0, 1, 2, 3]]
    i' = i + 1
    j' = j + i
    fas = [aux i' j' image dfa | image <- imgP]
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

{-joinMaps :: FiniteAutomata ImagePixels Int -> [FiniteAutomata ImagePixels Int] -> FiniteAutomata ImagePixels Int
joinMaps dfa [] = dfa
joinMaps initDfa (dfa:dfas) = joinMaps fa dfas
  where fa = FA {
-- unir initDfa con dfa
-}    

--recursiva primero una transicion y luego una lista de transiciones y cuando sea vacia retorna Map.Union Set.unions (Set.fromList $ List.map delta fas)  (Set.fromList $ List.map delta fas)

-- Como se cuentan elementos en una lista -- automata inicial unir el primero en la primera recursion



splitImg :: Int -> [[a]] -> [[a]]
splitImg _ xs | (length xs)==0 = xs | (length xs)==1 = xs
splitImg n xs | n==0 = Prelude.map (Prelude.take m) (Prelude.drop m xs) 
              | n==1 = Prelude.map (Prelude.take m) (Prelude.take m xs) 
              | n==2 = Prelude.map (Prelude.drop m) (Prelude.drop m xs) 
              | n==3 = Prelude.map (Prelude.drop m) (Prelude.take m xs) 
    where m = div (length xs) 2