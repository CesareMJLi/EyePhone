# EyePhone

EyePhone is a personal built project, aims to help blind people with their cellphone to detect the obstacles on the road. EyePhone focus on the common road scenes in the city. It has a quite simple use - “Hold the phone, whenever there is a obstacle in the road, remind the user.”

Obstacles could have following categrories: Stairs, Bicycle, Street Lights, Cars, Other People, and other obstacles.

There are some basic idea of how to build this babe:)

* Pure empirical method of CNN: We trust our CNN methods, and offer tons of data (input: images output: obstacles detected y/n) to train it, and later given specific input it could make a decision.

* Combine Machine Learning with Prior Knowledge: We have some prior knowledge in the obstacle detection, like when something is far it would be small in our sight, while close make it bigger in the sights.

* Combine CNN with RNN: This is a very promising idea while I have no idea of where to start now. I found one interesting fact occasionally, that in a badly scaled picture, sometimes it is hard even for human to find what’s in the picture. While it all the pictures are played fast, it would be easily found! (Imagine watching a badly scaled .gif picture!). So maybe we could combine CNN and RNN and contribute to a better performance. While it also seems that even if we are going to relate the temporal information with the static analysis, CNN and RNN may not be the only solutions. Here more information could be founded in my [blog](https://cesaremjli.github.io/2018/05/02/Papers-EyePhone/).

Currently lets start from selective search in a static pciture.
