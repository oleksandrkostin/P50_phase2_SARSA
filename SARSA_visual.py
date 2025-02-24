import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QGridLayout, QWidget, QInputDialog, QVBoxLayout, QComboBox, QDialog, QDialogButtonBox, QHBoxLayout, QSpacerItem, QSizePolicy
from PyQt5.QtGui import QColor, QPalette, QFont
from PyQt5.QtCore import Qt, QTimer
from SARSA import SPATIAL_RELATIONS, OBJECTS, set_goal, train

class SARSAVisualization(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SARSA Learning Visualization")

        # Set the main window background to black
        self.setStyleSheet("background-color: black;")

        # Create a central widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.grid_layout = QGridLayout()
        self.central_widget.setLayout(self.grid_layout)

        # Set a larger and bolder font for all labels
        self.font = QFont("Arial", 16, QFont.Bold)  # Increase font size to 16 and set to bold

        # Initialize grid labels
        self.labels = [[QLabel() for _ in range(5)] for _ in range(5)]
        for i in range(5):
            for j in range(5):
                label = self.labels[i][j]
                label.setAlignment(Qt.AlignCenter)
                label.setStyleSheet("background-color: black; color: green; border: 1px solid green;")  # Black background, green text
                label.setFont(self.font)  # Apply the larger and bolder font
                self.grid_layout.addWidget(label, i, j)

        # Add Y-axis labels (spatial relations) with a larger spacer to the left
        spacer = QSpacerItem(100, 20, QSizePolicy.Fixed, QSizePolicy.Fixed)  # Increased spacer width to 25
        for i, relation in enumerate(SPATIAL_RELATIONS):
            y_label = QLabel(relation)
            y_label.setStyleSheet("color: green;")  # Green text
            y_label.setFont(self.font)  # Apply the larger font
            self.grid_layout.addItem(spacer, i, 5)  # Add spacer to the left of the Y-axis labels
            self.grid_layout.addWidget(y_label, i, 6)  # Move Y-axis labels to column 6

        # Add X-axis labels (objects) with adjusted column width
        for j, obj in enumerate(OBJECTS):
            x_label = QLabel(obj)
            x_label.setAlignment(Qt.AlignCenter)
            x_label.setStyleSheet("color: green;")  # Green text
            x_label.setFont(self.font)  # Apply the larger font
            self.grid_layout.addWidget(x_label, 5, j)

        # Adjust column width to align X-axis labels with grid cells
        for j in range(5):
            self.grid_layout.setColumnMinimumWidth(j, 100)  # Set minimum column width

        # Maximize the window to full screen
        self.showMaximized()

    def update_grid(self, state, episode, total_reward, goal_reached=False):
        """Update the grid with the current state."""
        x, y = state
        for i in range(5):
            for j in range(5):
                label = self.labels[i][j]
                if i == x and j == y:
                    if goal_reached:
                        label.setStyleSheet("background-color: green; color: black; border: 1px solid green;")  # Green for goal
                    else:
                        label.setStyleSheet("background-color: red; color: black; border: 1px solid red;")  # Red for current state
                else:
                    label.setStyleSheet("background-color: black; color: green; border: 1px solid green;")  # Black background, green text
                label.setText(f"{SPATIAL_RELATIONS[i]} {OBJECTS[j]}")
        self.setWindowTitle(f"Episode {episode + 1} - Total Reward: {total_reward:.2f}")

def visualize_learning(Q_table, episode, state, total_reward, goal_reached=False):
    """Update the PyQt visualization."""
    app.processEvents()  # Process GUI events
    window.update_grid(state, episode, total_reward, goal_reached)

class GoalDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Set Goal")
        self.layout = QVBoxLayout()

        # Create a horizontal layout for the dropdowns
        dropdown_layout = QHBoxLayout()

        # Dropdown for spatial relations
        self.relation_combo = QComboBox()
        self.relation_combo.addItems(SPATIAL_RELATIONS)
        dropdown_layout.addWidget(QLabel("Spatial Relation:"))
        dropdown_layout.addWidget(self.relation_combo)

        # Dropdown for objects
        self.object_combo = QComboBox()
        self.object_combo.addItems(OBJECTS)
        dropdown_layout.addWidget(QLabel("Object:"))
        dropdown_layout.addWidget(self.object_combo)

        # Add the dropdown layout to the main layout
        self.layout.addLayout(dropdown_layout)

        # OK and Cancel buttons
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        self.layout.addWidget(self.button_box)

        self.setLayout(self.layout)

    def get_goal_position(self):
        """Return the selected goal position as a tuple (x, y)."""
        relation = self.relation_combo.currentText()
        obj = self.object_combo.currentText()
        return (SPATIAL_RELATIONS.index(relation), OBJECTS.index(obj))

def set_goal_pyqt():
    """Set the goal position using a custom PyQt dialog."""
    dialog = GoalDialog()
    if dialog.exec_() == QDialog.Accepted:
        goal_position = dialog.get_goal_position()
        print(f"Goal set: {SPATIAL_RELATIONS[goal_position[0]]} {OBJECTS[goal_position[1]]} -> {goal_position}")
        return goal_position
    else:
        print("Goal setting canceled. Using default goal.")
        return (0, 0)  # Default goal

def main():
    global app, window

    # Ask the user if they want to enable visualization
    visualize_choice = input("Do you want to enable visualization? (yes/no): ").strip().lower()
    visualize = visualize_choice == "yes" or visualize_choice == "y"

    # Initialize PyQt application
    app = QApplication(sys.argv)
    window = SARSAVisualization()

    if visualize:
        # Set the goal using PyQt input dialog
        goal_position = set_goal_pyqt()
        window.show()  # Show the PyQt window
        reward_history = train(visualize_callback=visualize_learning, new_goal_position=goal_position)
    else:
        # Set the goal using console input
        set_goal()
        reward_history = train(visualize_callback=None)

    # Plot the learning progress after training
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(reward_history)
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("SARSA Learning Progress")
    plt.show()

    # Quit the PyQt application after the matplotlib window is closed
    app.quit()

if __name__ == "__main__":
    main()