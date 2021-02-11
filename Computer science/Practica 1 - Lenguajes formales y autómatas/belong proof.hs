module ElementBelongsTo where

import Data.List as List
import Data.Set as Set
import Data.Map as Map

elementBelongsTo::(Eq x) => x-> [x]-> Bool
elementBelongsTo x [] = False
elementBelongsTo x (y : ys)
  | x == y    = True
  | otherwise =  x `elementBelongsTo` ys
