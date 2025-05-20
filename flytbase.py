import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Waypoint tuple that has coordinates and distance to a point
Waypoint = lambda x, y, z: type('Waypoint', (), {'x':x, 'y':y, 'z':z, 
                     'distance_to': lambda self, other: np.sqrt((self.x-other.x)**2 + 
                                                               (self.y-other.y)**2 + 
                                                               (self.z-other.z)**2)})()

# Drone Trajectory Class for each Drone
class DroneTrajectory:
    
    def __init__(self, drone_id, waypoints, time_range, color=None):
        self.drone_id = drone_id
        self.waypoints = waypoints
        self.t_start, self.t_end = time_range
        self.color = color 
        self.total_distance = 0

        for i in range(len(self.waypoints)-1):
            self.total_distance+= self.waypoints[i].distance_to(self.waypoints[i+1])

        
    def position_const_timesegment(self, t):
        # To calculated position of Drone at instant t

        # t not in time period
        if not self.t_start <= t <= self.t_end:
            return None
            
        # Identify which Segment drone is in
        n_segments = len(self.waypoints) - 1
        seg_idx = min(int((t - self.t_start) / (self.t_end - self.t_start) * n_segments), n_segments - 1)

        # Identify progress in that particular segment
        progress = (t - (self.t_start + seg_idx*(self.t_end-self.t_start)/n_segments))/ ((self.t_end-self.t_start)/n_segments)
        
        # Use the vector direction along the waypoints to interpolate
        wp1, wp2 = self.waypoints[seg_idx], self.waypoints[seg_idx+1]
        return Waypoint(wp1.x + progress*(wp2.x-wp1.x),
                        wp1.y + progress*(wp2.y-wp1.y),
                        wp1.z + progress*(wp2.z-wp1.z))
    

    def position_const_velocity(self, t):
        # Calculate Required velocity
        vel = self.total_distance / (self.t_end - self.t_start)
        
        # Check if time is outside mission window
        if not self.t_start <= t <= self.t_end:
            return None
        
        # Calculate total distance traveled by time t
        distance_traveled = vel * (t - self.t_start)
        
        cur_dist = 0
        segment_start_time = self.t_start
        
        # Find which segment the drone is in at time t
        for i in range(len(self.waypoints)-1):
            segment_distance = self.waypoints[i].distance_to(self.waypoints[i+1])
            segment_time = segment_distance / vel
            
            # Check if we're in this segment
            if distance_traveled <= cur_dist + segment_distance:
                # Calculate how far along this segment we are
                segment_progress = (distance_traveled - cur_dist) / segment_distance
                
                # Interpolate position along this segment
                x = self.waypoints[i].x + segment_progress * (self.waypoints[i+1].x - self.waypoints[i].x)
                y = self.waypoints[i].y + segment_progress * (self.waypoints[i+1].y - self.waypoints[i].y)
                z = self.waypoints[i].z + segment_progress * (self.waypoints[i+1].z - self.waypoints[i].z)
                
                return Waypoint(x, y, z)
            
            cur_dist += segment_distance
            segment_start_time += segment_time
        
        return self.waypoints[-1]



# To create the 4D plot
def create_animation(trajectories, primary_id, safety_dist=15, duration=5, fps=10):
    
    fig = plt.figure(figsize=(15, 8))
    ax_3d = fig.add_subplot(121, projection='3d')
    ax_xz = fig.add_subplot(122)
    drone_colors = {traj.drone_id: traj.color for traj in trajectories}

    # XZ plot is displayed to show that there may look like there is another drone in the circle. 
    # However in 3D dimensions, the other drone is not within the sphere

    # Utilize all given point of travel for axis limits
    all_wps = [wp for traj in trajectories for wp in traj.waypoints]
    x_min, x_max = min(wp.x for wp in all_wps)-10, max(wp.x for wp in all_wps)+10
    y_min, y_max = min(wp.y for wp in all_wps)-10, max(wp.y for wp in all_wps)+10
    z_min, z_max = min(wp.z for wp in all_wps)-10, max(wp.z for wp in all_wps)+10

    # Select the primary drone
    primary = next(t for t in trajectories if t.drone_id == primary_id)

    # Pre-calculate all time points and conflicts
    total_frames = int(fps * duration)
    frame_times = np.linspace(primary.t_start, primary.t_end, total_frames)
    conflicts = [] 
        
    # Track conflict states
    previous_conflict_drones = set()
    current_conflict_number = 0
    conflict_active = False

    # Pre-compute conflicts at each frame time
    for i, t in enumerate(frame_times):
        pos1 = primary.position_const_timesegment(t)
        if not pos1: 
            conflicts.append(None)
            continue
        
        time_conflicts = []
        current_drones_in_conflict = set()
        
        for traj in trajectories:
            # Ignore the primary drone
            if traj.drone_id == primary_id: 
                continue
                
            pos2 = traj.position_const_timesegment(t)
            
            # Check for conflict
            if pos2 and pos1.distance_to(pos2) < safety_dist:
                time_conflicts.append({
                    'time': t, 
                    'pos': pos1, 
                    'dist': pos1.distance_to(pos2), 
                    'droneid': traj.drone_id,
                    'frame': i  # Store frame index for precise matching
                })
                current_drones_in_conflict.add(traj.drone_id)
        
        # Check if conflict state has changed
        if current_drones_in_conflict:
            if current_drones_in_conflict != previous_conflict_drones:
                # New conflict pattern of drones detected
                current_conflict_number += 1
                conflict_active = True
                print(f"\nConflict #{current_conflict_number} detected at frame {i} (time {t:.1f}s) with drones: {', '.join(current_drones_in_conflict)}")
                
                # Detail of conflict pattern
                for conflict in time_conflicts:
                    print(f"  - Drone {conflict['droneid']} at distance {conflict['dist']:.2f} units")
                
                previous_conflict_drones = current_drones_in_conflict.copy()
        else:
            if conflict_active:
                # Conflict Pattern has ended
                print(f"Conflict #{current_conflict_number} resolved at frame {i} (time {t:.1f}s)")
                conflict_active = False
                previous_conflict_drones = set()
        
        # Store conflicts for visualization
        conflicts.append({
            'time': t,
            'conflicts': time_conflicts,
            'conflict_number': current_conflict_number if time_conflicts else None,
            'frame': i
        })

    # Print clear if no conflicts were detected
    if current_conflict_number == 0:
        print("\nClear - No conflicts detected during the entire mission")

    # Animation update function
    def update(frame):

        ax_3d.clear()
        ax_xz.clear()
        t = frame_times[frame]  # Use precomputed time for this frame
        
        # Set axis limits for both views
        for ax, lims in [(ax_3d, (x_min, x_max, y_min, y_max, z_min, z_max)), 
                         (ax_xz, (x_min, x_max, z_min, z_max))]:
            ax.set_xlim(lims[0], lims[1])
            ax.set_ylim(lims[2], lims[3])
            if ax == ax_3d: 
                ax.set_zlim(lims[4], lims[5])
        
        legend_handles = []

        # Plot all trajectories
        for traj in trajectories:
            xs, ys, zs = zip(*[(wp.x, wp.y, wp.z) for wp in traj.waypoints])
            
            # Plot trajectory lines
            ax_3d.plot(xs, ys, zs, 'o-', color=traj.color, alpha=0.1)
            ax_xz.plot(xs, zs, 'o-', color=traj.color, alpha=0.1)
            
            # Plot current position
            pos = traj.position_const_timesegment(t)
            if pos:
                ax_3d.scatter(pos.x, pos.y, pos.z, color=traj.color, s=100)
                ax_xz.scatter(pos.x, pos.z, color=traj.color, s=100)
                
                # Add safety sphere for primary drone
                if traj.drone_id == primary_id:

                    # Stores Position of Primary Drone
                    primary_pos = pos
                    u, v = np.mgrid[0:2*np.pi:15j, 0:np.pi:15j]
                    ax_3d.plot_surface(
                        pos.x + safety_dist*np.cos(u)*np.sin(v),
                        pos.y + safety_dist*np.sin(u)*np.sin(v),
                        pos.z + safety_dist*np.cos(v),
                        color='blue', alpha=0.1
                    )
                    ax_xz.add_patch(plt.Circle(
                        (pos.x, pos.z), safety_dist, 
                        color='blue', alpha=0.1
                    ))
            
            # Add to legend
            line_3d = ax_3d.plot(xs, ys, zs, 'o-', color=traj.color, alpha=0.1, label=traj.drone_id)[0]
            legend_handles.append(line_3d)
        
        ax_3d.legend(handles=legend_handles, loc='upper right')
        
        # Check for conflicts in the frame
        frame_conflicts = conflicts[frame]['conflicts'] if conflicts[frame] else None
        
        if frame_conflicts:
            # Highlights conflict with cross
            
            ax_3d.scatter(
                primary_pos.x, primary_pos.y, primary_pos.z, 
                color='red', s=200, marker='x'
            )
            ax_xz.scatter(
                primary_pos.x, primary_pos.z, 
                color='red', s=200, marker='x'
            )
            
            # Create conflict message
            conflict_messages = [
                f" {c['droneid']} (Distance: {c['dist']:.2f})" 
                for c in frame_conflicts
            ]
            fig.suptitle(
                f"CONFLICT {conflicts[frame]['conflict_number']} with:\n" + 
                "\n".join(conflict_messages) + 
                f"\nTime: {t:.1f}s", 
                color='red'
            )
        else:
            fig.suptitle(f"Time: {t:.1f}s (Safe)", color='green')
                
        # Set axis labels
        ax_3d.set_xlabel('X')
        ax_3d.set_ylabel('Y')
        ax_3d.set_zlabel('Z')
        ax_xz.set_xlabel('X')
        ax_xz.set_ylabel('Z')
        ax_xz.set_title('XZ Projection')
        ax_xz.grid(True)
    
    # Create animation with proper timing (1000ms / fps = interval in ms)
    return FuncAnimation(
        fig, 
        update, 
        frames=total_frames, 
        interval=1000/fps,  # Correct interval for requested FPS
        repeat=False
    )

if __name__ == "__main__":

    # List of each drone waypoints and time period

    # Conflict Scenario 1
    # trajectories = [
    #     DroneTrajectory("drone1", [Waypoint(0,0,0), Waypoint(50,50,40), Waypoint(100,0,80)], (0,60), "red"),
    #     DroneTrajectory("drone2", [Waypoint(0,100,100), Waypoint(50,50,20), Waypoint(100,100,0)], (20,80), "green"),
    #     DroneTrajectory("drone3", [Waypoint(100,100,100), Waypoint(70,100,20), Waypoint(0,100,0)], (40,100), "purple"),
    #     DroneTrajectory("drone4", [Waypoint(100,0,0), Waypoint(60,60,10), Waypoint(0,100,0)], (50,110), "orange"),
    #     DroneTrajectory("primary", [Waypoint(0,50,20), Waypoint(50,50,20), Waypoint(70,70,20),Waypoint(60,60,10), Waypoint(100,50,0)], (30,90), "blue")
    # ]

    # #  Conflict Scenario 2
    # trajectories = [    
    # DroneTrajectory("primary",[Waypoint(0, 0, 0),Waypoint(100, 100, 50),Waypoint(200, 0, 100)],(0, 60), "blue"),
    # DroneTrajectory("drone1", [Waypoint(200, 0, 0),Waypoint(100, 100, 50),Waypoint(0, 200, 100)],(10, 50), "red"),
    # DroneTrajectory("drone2", [Waypoint(50, 0, 0),Waypoint(50, 90, 60), Waypoint(50, 200, 120)],(10, 45), "green")
    # # Near miss on Drone 2
    # ]


    #  No Conflict
    trajectories = [
    DroneTrajectory("primary", [Waypoint(0, 50, 10), Waypoint(100, 50, 10),Waypoint(200, 50, 10)],  (0, 60), "blue"),
    DroneTrajectory("drone2", [Waypoint(0, 100, 10),Waypoint(100, 100, 10), Waypoint(200, 100, 10)], (0, 60), "red"),
    DroneTrajectory("drone3",  [Waypoint(100, 0, 50), Waypoint(100, 200, 50)], (20, 40), "green"),
    DroneTrajectory("drone4", [Waypoint(50, 50, 10), Waypoint(150, 150, 10)], (70, 90), "orange")
    ]


    anim = create_animation(trajectories, "primary")
    plt.show()
    