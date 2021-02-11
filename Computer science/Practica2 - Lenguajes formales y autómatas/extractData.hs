{- | The following is a program that takes a XML file, containing all pages
in the 20-th century mathematicians category from Wikipedia, extracts relevant
data from those mathematicians (name, birth and death date, birth and death
place, alma mater and fields of study) and creates a CSV file that contains
that information.-}
module ExtractData where

import Text.Regex.TDFA
import Data.Strings
import System.IO
import Data.CSV

-- | This is the primary function that reads the XML file, extracts the data
-- and creates the CSV file.
extractData :: IO ()
extractData = do
    w'' <- openFile "Wikipedia.xml" ReadMode
    w'  <- hGetContents w''
    let w  = strSplitAll "<page>" w'
    let m  = [d] ++ [vector page | page <- init (tail w)]
    let f  = genCsvFile m
    writeFile "Data.csv" f
    where
      d = ["Name","Birth date","Death date","Birth place",
                 "Death place","Alma mater","Fields"]

-- | Creates a list containing all information taken from the infobox of each
-- Wikipedia article.
vector :: String -> [String]
vector page = [nm, bd, dd, bp, dp, am, fs]
  where
    nm = names n "<title>(.*)</title>" !! 0
    bd = if length z > 0
         then clean' ((dates z) !! 0) "[]{}\n"
         else ""
    dd = if length y > 0
         then clean' ((dates y) !! 0) "[]{}\n"
         else ""
    bp = if length (names n "# *birth_place *= *([^#]*)") > 0
         then
           (clean' ((names n "# *birth_place *= *([^#]*)") !! 0) "[]")
         else ""
    dp = if length (names n "# *death_place *= *([^#]*)") > 0
         then
           (clean' ((names n "# *death_place *= *([^#]*)") !! 0) "[]")
         else ""
    am = if length (names n "# *alma_mater *= *([^#]*)") > 0
         then clean' ((names n "# *alma_mater *= *([^#]*)") !! 0)  "[]"
         else ""
    fs = if length (names n "# *fields? *= *([^#]*)") > 0
         then clean' ((names n "# *fields? *= *([^#]*)") !! 0)     "[]"
         else ""
    y    = names n "# *death_date *= *JAMER([^#]*#[^#]*#[^#]#[^#]*).*REMAJ"
    z    = names n "# *birth_date *= *JAMER([^#]*#[^#]*#[^#]#[^#]*).*REMAJ"
    l    = [strReplace "|" "#"  x | x <- lines page]
    m    = [strReplace "{{" "JAMER" x | x <- l]
    n    = [strReplace "}}" "REMAJ" x | x <- m]
    bp'  = strReplace "JAMER" "" bp
    dp'  = strReplace "JAMER" "" dp
    bp'' = strReplace "REMAJ" "" bp'
    dp'' = strReplace "REMAJ" "" dp'

-- | Takes the year, month and day from the date value contained into the
-- infobox of a Wikipedia page, if it has got one.
dates :: [String] -> [String]
dates [] = []
dates b  = [listToString (y ++ ["/"] ++ m ++ ["/"] ++ d)]
    where
        y = [listToString (names b "[^#]*#([^#]*)#[^#]*#[^#]*")]
        m = [listToString (names b "[^#]*#[^#]*#([^#]*)#[^#]*")]
        d = [listToString (names b "[^#]*#[^#]*#[^#]*#([^#]*)")]

-- | Searches for a String contained into another one as a regular expression.
names :: [String] -> String -> [String]
names [] _ = []
names (x:xs) m
  | x =~ m = [(((x =~ m :: [[String]]) !! 0) !! 1)] ++ (names xs m)
  | otherwise = names xs m

-- | Takes a list of Strings and converts it into a single String.
-- In this case, it recieves a list resulting from the Prelude.words
-- function we previously used in 'extractData' to remove line breaks from
-- the XML file.
listToString :: [String] -> String
listToString []     = ""
listToString [x]    = x
listToString (x:xs) = x ++ "\n" ++ (listToString xs)

-- | Removes a symbol from a String in all of its occurrences.
clean :: String -> Char -> String
clean "" c    = ""
clean (x:xs) c
  | x == c    = clean xs c
  | otherwise = [x] ++ (clean xs c)

-- | Removes every symbol of a String from another one in all of its ocurrences.
clean' :: String -> String -> String
clean' s []     = s
clean' s [x]    = clean s x
clean' s (x:xs) = clean' (clean s x) xs