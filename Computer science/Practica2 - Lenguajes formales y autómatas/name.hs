module Vector where
import Data.List.Split
import Text.Regex.TDFA

--vector :: String -> [String]
--vector page = [lf, cl, ca, ct, tf]
--    where
--       lf = (client n "<lobbying_firm>(.*)</lobbying_firm>") !! 0
--       cl = if length (client n "<client_last_name>(.*)</client_last_name>") > 0
--            then
--              (client n "<client_last_name>(.*)</client_last_name>" !! 0)
--            else ""
--       ca= "re"
--       ct= "re"
--       tf="re"
--       n = [page] ::[String]
vector :: String -> [String]
vector page = [lf, cl, ca, ct, tf]
    where
       lf = if length (client n "<lobbying_firm>(.*)</lobbying_firm>") > 0
            then
            (client n "<lobbying_firm>(.*)</lobbying_firm>" !! 0)
            else ""
       cl = if length (client n "<client_last_name>(.*)</client_last_name>") > 0
            then
               (client n "<client_last_name>(.*)</client_last_name>" !! 0)
            else ""
       ca = if length (client n "re") > 0
            then
              (client n "re" !! 0)
           else ""
       ct = if length (client n "re") > 0
            then
              (client n "re" !! 0)
            else ""
       tf = if length (client n "re") > 0
            then
              (client n "re" !! 0)
            else ""
       n = [page] ::[String]


client :: [String] -> String -> [String]
client [] _ = []
client (x:xs) m
  | x =~ m = [(((x =~ m :: [[String]]) !! 0) !! 1)] ++ (client xs m)
  | otherwise = client    xs m

    