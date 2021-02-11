{- |
Module       :  ConstructAutomaton
Description  :  this module is used to create a finite automata and generate an
                image according to this
Copyright    :  Hamilton Smith Gómez Osorio, 2019
                Pablo Alberto Osorio Marulanda,  2019
-}
module ConstructAutomaton where

import FiniteAutomata
import Data.Set as Set
import Data.Map as Map
import Data.List as List
import Codec.Picture

{- |  Returns the respective Deterministic Finite Automaton of an image.
      Receives the pixel representation of an image and return a DFA with an
      ImagePixels which represents the states and an
      integer as its symbols. -}
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
        } where
            delta' :: TransitionFunction state symbol
            delta'= Map.empty


{- |  Determines the necessary iterations to create the respective DFA
      Receives an ImagePixels that represents the initial state and according
      to its length defines the additional iterations to be carried out,
      of integer type, with respect to the base case of (2 ^ 3) x (2 ^ 3)
      pixels to create the respective DFA of the image. -}
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


{- |  Defines the structure of a Deterministic Finite Automaton
      This fuction takes three integers, two ImagePixels and a dfa.
      Its fuction is to implement the algorithm, reiterating the i and j
      integer values, which represent state where the algorith is and the
      states that the algorithm has visited.
      This fuction define, in a recursive way, all the dfa's that an image
      is able to create. Those are saved in "fas" and they are sorted by the
      joinAu fuction.. -}
aux :: Int -> Int -> ImagePixels -> ImagePixels ->
  FiniteAutomata ImagePixels Int -> FiniteAutomata ImagePixels Int
aux i j img img2 dfa@(FA st sy tf as es) = if i==j && (i/=0&&j/=0)
   then dfa
 else u
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
    imgP :: [ImagePixels]
    imgP = [splitList img k| k <- [0, 1, 2, 3]]
    i' :: Int
    i' = i + 1
    j' :: Int
    j' = j + (i-scaleDiference img2)
    fas :: [FiniteAutomata ImagePixels Int]
    fas = [aux i' j' image img2 dfa | image <- imgP]
    u :: FiniteAutomata ImagePixels Int
    u = joinAu fas


{- |  Sorts the automaton created by the processed images
      Receives a list of finite automata, corresponding to automatatas
      resulting from dividing the image that were processed by the aux function.
      This fuction sorts all the states, sybols and transitions created in
      the last fuction, it defines all the parameter of a final dfa -}
joinAu :: [FiniteAutomata ImagePixels Int] -> FiniteAutomata ImagePixels Int
joinAu fas = FA{
      states = Set.unions (List.map states fas)
        , alphabet  = imageAlphabet
        , delta = delta1 (delta (head fas)) (tail (List.map delta (tail fas)))
        , initialState = initialState (head fas)
        , acceptState  = Set.unions (List.map acceptState fas)
      }
      where
        delta1 :: TransitionFunction ImagePixels Int ->
           [TransitionFunction ImagePixels Int] ->
            TransitionFunction ImagePixels Int
        delta1 tf [] = tf
        delta1 d tfs = Map.union d (delta1 (head tfs) (tail tfs))


{- |  Processes pixel coordinates
      Receives the coordinates of a pixel in an image and assign a password
      corresponding to the position occupied by the pixel in the subdivisions
      of the image made by the automaton. -}
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
    s :: [Int]
    s = []
    l :: Int
    l = (n `div` 2)
    a :: Int
    a = (m `div` 2)


{- |  Determines the transitions map
      This fuction takes a dfa and a ImagePixels to return , through the
      transitions , the associated map of transitions. -}
return1Map :: FiniteAutomata ImagePixels Int -> ImagePixels ->
  Map Int (Set ImagePixels)
return1Map dfa img = (delta dfa)!img


{- |  Finds and ImagePixel
      This fuction finds, recursively , the ImagePixel associated to a
      password. -}
returnPixel4 :: [Int] -> Map Int(Set ImagePixels) ->
   FiniteAutomata ImagePixels Int ->ImagePixels
returnPixel4 ad m dfa
  | length ad==1 = Set.elemAt 0 (m!(head ad))
  | otherwise = returnPixel4 (tail ad)
  (return1Map dfa (Set.elemAt 0 (m!(head ad)))) dfa


{- |  Identifies associated pixel from a password
      This fuction returns a Pixel linked to a especific pair of coordinates
      using the returnPixel4 function and the transition fuction. -}
returnPixel2 :: Int -> Int -> FiniteAutomata ImagePixels Int -> Int ->
   PixelRGB8
returnPixel2 x y dfa n = ((returnPixel4 (coords x y n n) fMap dfa)!!0)!!0
 where
   fMap :: Map Int (Set ImagePixels)
   fMap = return1Map dfa (initialState dfa)


{- |  Creates an image
      This fuction takes a specific resolution and a defined dfa to use the
      "generateImage" fuction, which return a dynamic image in a png format
      from the juicyPixels package. -}
createImage :: Int -> FiniteAutomata ImagePixels Int -> DynamicImage
createImage n dfa =
   ImageRGB8(generateImage (\x y -> returnPixel2 x y dfa (2^n) ) (2^n) (2^n))


{- |  Splits an ImagePixels
      Receives an ImagePixels corresponding to a state and an integer to
      return its subdivision associated to the integer. -}
splitList ::  ImagePixels -> Int -> ImagePixels
splitList l a
  | length l == 1 = l
  | a==1 = [Prelude.take (length l `div` 2) x | x <- lT]
  | a==3 = [Prelude.drop (length l `div` 2) x | x <- lT]
  | a==0 = [Prelude.take (length l `div` 2) x | x <- lD]
  | a==2 = [Prelude.drop (length l `div` 2) x | x <- lD]
      where
       lT :: ImagePixels
       lT= Prelude.take (length l `div` 2) l
       lD :: ImagePixels
       lD= Prelude.drop (length l `div` 2) l


