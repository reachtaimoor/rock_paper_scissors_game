import tkinter as tk

class RockPaperScissorsGame:
    def __init__(self, master):
        self.master = master
        master.title("2 Player Rock Paper Scissors Game")

        # Initialize scores
        self.player1_score = 0
        self.player2_score = 0

        # Create frames for each player
        self.frame1 = tk.Frame(master, width=300, height=300)
        self.frame1.pack(side=tk.LEFT, padx=10, pady=10)

        self.frame2 = tk.Frame(master, width=300, height=300)
        self.frame2.pack(side=tk.RIGHT, padx=10, pady=10)

        # Create score labels
        self.score_label1 = tk.Label(self.frame1, text="Player 1 Score: 0", font=("Arial", 14))
        self.score_label1.pack(pady=10)

        self.score_label2 = tk.Label(self.frame2, text="Player 2 Score: 0", font=("Arial", 14))
        self.score_label2.pack(pady=10)

        # Create choice buttons for Player 1
        self.player1_choice = tk.StringVar(value="")
        self.create_choice_buttons(self.frame1, self.player1_choice, "Player 1")

        # Create choice buttons for Player 2
        self.player2_choice = tk.StringVar(value="")
        self.create_choice_buttons(self.frame2, self.player2_choice, "Player 2")

        # Create a button to determine the winner
        self.result_button = tk.Button(master, text="Determine Winner", command=self.determine_winner)
        self.result_button.pack(pady=20)

        # Result label
        self.result_label = tk.Label(master, text="", font=("Arial", 16))
        self.result_label.pack(pady=10)

    def create_choice_buttons(self, frame, player_choice, player_name):
        tk.Label(frame, text=f"{player_name}, choose your option:", font=("Arial", 12)).pack(pady=10)
        tk.Radiobutton(frame, text="Rock", variable=player_choice, value="rock").pack(anchor=tk.W)
        tk.Radiobutton(frame, text="Paper", variable=player_choice, value="paper").pack(anchor=tk.W)
        tk.Radiobutton(frame, text="Scissors", variable=player_choice, value="scissors").pack(anchor=tk.W)

    def determine_winner(self):
        p1_choice = self.player1_choice.get()
        p2_choice = self.player2_choice.get()

        if p1_choice == "" or p2_choice == "":
            self.result_label.config(text="Both players must make a choice!")
            return

        if p1_choice == p2_choice:
            self.result_label.config(text="It's a tie!")
        elif (p1_choice == "rock" and p2_choice == "scissors") or \
             (p1_choice == "scissors" and p2_choice == "paper") or \
             (p1_choice == "paper" and p2_choice == "rock"):
            self.result_label.config(text="Player 1 wins!")
            self.player1_score += 1
        else:
            self.result_label.config(text="Player 2 wins!")
            self.player2_score += 1

        # Update scores
        self.score_label1.config(text=f"Player 1 Score: {self.player1_score}")
        self.score_label2.config(text=f"Player 2 Score: {self.player2_score}")

if __name__ == "__main__":
    root = tk.Tk()
    game = RockPaperScissorsGame(root)
    root.mainloop()