                        mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                    )
                    
                    # Display angles on image
                    cv2.putText(image, f"Elbow: {int(elbow_angle)}", 
                                (int(elbow[0] * image.shape[1]), int(elbow[1] * image.shape[0])), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(image, f"Knee: {int(knee_angle)}", 
                                (int(knee[0] * image.shape[1]), int(knee[1] * image.shape[0])), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                    
                except Exception as e:
                    pass  # If landmarks are not detected correctly
            
            # Update counters in UI
            bicep_count_text.markdown(f"**Bicep Curls:** {st.session_state.bicep_counter.count}")
            squat_count_text.markdown(f"**Squats:** {st.session_state.squat_counter.count}")
            pushup_count_text.markdown(f"**Pushups:** {st.session_state.pushup_counter.count}")
            
            # Update recommendations
            recommendations = recommend_exercises(current_exercise)
            if current_exercise != "unknown":
                recommendations_text.markdown(f"**Current Exercise:** {current_exercise.title()}\n\n**Try Next:** {', '.join(recommendations)}")
            else:
                recommendations_text.markdown("Get into position to start an exercise!")
            
            # Display image
            stframe.image(image, channels="BGR", use_column_width=True)
            
            # Check if the Streamlit app is still running
            if not cap.isOpened():
                break
                
            # Small sleep to reduce CPU usage
            time.sleep(0.01)
    
    cap.release()

if __name__ == "__main__":
    main()
