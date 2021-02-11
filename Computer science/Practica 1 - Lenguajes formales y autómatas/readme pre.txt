ReadME

Name of the program: Representing Images with Deterministic Finite Automata

General description of the program:

This is program which was written in the fuctional programming language Haskell. It means,
all the program itself is defined by different functions which let us take an image, make 
a Determistic Finite Automata which reads that image and then generates the image again 
with a specific resolution.

How to use it:

To use the program you will need, firstable, a list of list of pixels, which the program 
defines as "ImagePixels" type. 
Then use de "img2dfa" fuction to generate a deterministic finite automata linked to that image.
In order to create the image again use the fuction "createImage" which take an integer 
parameter (resolution of the new image) and the dfa that the program created before.
Now, we use the "savePngImage" command to save the created image in a png format.
If you are processing images with many resolutions pixels, let the promman take its time

For example:
(console interface)
      GHCi> cb = img2dfa chessBoard
      GHCi> cb_img = createImage 7 cb
      GHCi> savePngImage "chessBoard.png" cb_img

Fuctions' description


img2dfa::ImagePixels -> FiniteAutomata ImagePixels Int
This is one of the principal fuctions of the program, which takes an ImagePixels and return 
the dfa asociated to that image.


ScaleDiference::ImagePixels->Int
This fuction takes a selected image and returns an integer parameter to use in the algorithm
process.

aux :: Int -> Int -> ImagePixels -> ImagePixels -> FiniteAutomata ImagePixels Int -> FiniteAutomata ImagePixels Int

This fuction takes 3 integers, 2 ImagePixels and a dfa. Its fuction is to implement the algorithm, reiterating the 
i and j integer values, which represent state where the algorith is and the states that the algorithm has visited.
This fuction define, in a recursive way, all the dfa's that an image is able to create. Those are saved in "fas" and they
are sorted by the joinAu fuction.

joinAu :: [FiniteAutomata ImagePixels Int] -> FiniteAutomata ImagePixels Int
This fuction sorts all the states, sybols and transitions created in the last fuction, it defines all the parameter of a final dfa.


return1Map :: FiniteAutomata ImagePixels Int -> ImagePixels -> Map Int (Set ImagePixels)
This fuction takes a dfa and a ImagePixels to return , through de transitions , the associated
map of transitions.

returnPixel4 :: [Int] -> Map Int(Set ImagePixels) -> FiniteAutomata ImagePixels Int ->ImagePixels
This fuction finds, recursively , the ImagePixel associated to a path or password.

returnPixel2 :: Int -> Int -> FiniteAutomata ImagePixels Int -> Int -> PixelRGB8
This fuction returns a Pixel linked to a especific pair of coordinates using the last methods
and the transition fuction.

createImage :: Int -> FiniteAutomata ImagePixels Int -> DynamicImage
This fuction is another of the principal ones. This one takes a specific resolution and a defined dfa 
to use the "generateImage" fuction, which return a dynamic image in a png format.




 

